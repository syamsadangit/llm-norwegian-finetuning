from __future__ import annotations

from pathlib import Path
import logging
import sys
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from config_loader import load_config

CFG = load_config(BASE_DIR / "INPUT")

SEED = int(CFG["SEED"])
MAX_LEN = int(CFG["MAX_LEN"])

DATA_PATH = BASE_DIR / str(CFG["DATA_PATH"])
MODEL_NAME = str(CFG["BASE_MODEL"])

TRAIN_SUBSET_SIZE = int(CFG["TRAIN_SUBSET_SIZE"])
NUM_EPOCHS = int(CFG["NUM_EPOCHS"])
BATCH_SIZE = int(CFG["BATCH_SIZE"])
GRAD_ACCUM = int(CFG["GRAD_ACCUM"])
LEARNING_RATE = float(CFG["LEARNING_RATE"])
LOGGING_STEPS = int(CFG["LOGGING_STEPS"])
SAVE_STEPS = int(CFG["SAVE_STEPS"])
SAVE_TOTAL_LIMIT = int(CFG["SAVE_TOTAL_LIMIT"])

REPORTS_DIR = BASE_DIR / str(CFG.get("REPORTS_DIR", "reports"))
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = BASE_DIR / str(CFG.get("MODELS_DIR", "models"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODELS_PREFIX = str(CFG.get("MODELS_PREFIX", "distilgpt2-no"))


def get_next_run_id(reports_dir: Path, prefix: str = "train") -> int:
    """
    Finds the next available run id based on existing:
      - reports/train_N.log
      - reports/train_N/ (dir)
    """
    nums: List[int] = []

    for p in reports_dir.glob(f"{prefix}_*.log"):
        m = re.search(rf"{re.escape(prefix)}_(\d+)\.log$", p.name)
        if m:
            nums.append(int(m.group(1)))

    for p in reports_dir.glob(f"{prefix}_*"):
        if p.is_dir():
            m = re.search(rf"{re.escape(prefix)}_(\d+)$", p.name)
            if m:
                nums.append(int(m.group(1)))

    return (max(nums) + 1) if nums else 1


def setup_logging(log_file: Path) -> None:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logging.info(f"Log file: {log_file}")


def find_latest_checkpoint(model_dir: Path) -> Optional[Path]:
    """
    Finds the latest checkpoint in OUT_DIR (checkpoint-XXXX),
    choosing the one with the largest step number.
    """
    if not model_dir.exists():
        return None

    checkpoints = []
    for p in model_dir.glob("checkpoint-*"):
        if p.is_dir():
            m = re.search(r"checkpoint-(\d+)$", p.name)
            if m:
                checkpoints.append((int(m.group(1)), p))

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


def get_next_model_dir(models_dir: Path, prefix: str) -> Path:
    """
    Creates a new model directory per run:
      models/<prefix>_1, models/<prefix>_2, ...
    """
    nums: List[int] = []
    for p in models_dir.glob(f"{prefix}_*"):
        if p.is_dir():
            m = re.search(rf"{re.escape(prefix)}_(\d+)$", p.name)
            if m:
                nums.append(int(m.group(1)))

    next_n = (max(nums) + 1) if nums else 1
    return models_dir / f"{prefix}_{next_n}"


@dataclass
class MetricStore:
    steps: List[int] = field(default_factory=list)
    loss: List[float] = field(default_factory=list)
    learning_rate: List[float] = field(default_factory=list)
    grad_norm: List[float] = field(default_factory=list)

    def add(self, step: int, logs: Dict[str, Any]) -> None:
        self.steps.append(step)
        self.loss.append(float(logs.get("loss")) if "loss" in logs else float("nan"))
        self.learning_rate.append(
            float(logs.get("learning_rate")) if "learning_rate" in logs else float("nan")
        )
        self.grad_norm.append(
            float(logs.get("grad_norm")) if "grad_norm" in logs else float("nan")
        )


class MetricsLoggerCallback(TrainerCallback):
    def __init__(self, store: MetricStore):
        self.store = store

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        self.store.add(state.global_step, logs)

        ordered_keys = ["loss", "learning_rate", "grad_norm", "epoch"]
        parts = []
        for k in ordered_keys:
            if k in logs and isinstance(logs[k], (int, float)):
                parts.append(f"{k}={logs[k]:.6f}")
        for k, v in logs.items():
            if k in ordered_keys:
                continue
            if isinstance(v, (int, float)):
                parts.append(f"{k}={v:.6f}")
        if parts:
            logging.info(f"step={state.global_step} | " + " | ".join(parts))


def save_plots(store: MetricStore, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    import math

    def has_any_finite(vals: List[float]) -> bool:
        return any((v is not None) and (not math.isnan(v)) for v in vals)

    x = store.steps

    def plot_series(y: List[float], title: str, ylabel: str, filename: str) -> None:
        if not x or not has_any_finite(y):
            logging.info(f"Skipping plot {filename} (no data).")
            return

        plt.figure()
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel("Step")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=200)
        plt.close()

    plot_series(store.loss, "Training loss vs step", "Loss", "loss.png")
    plot_series(store.learning_rate, "Learning rate vs step", "Learning rate", "learning_rate.png")
    plot_series(store.grad_norm, "Gradient norm vs step", "Grad norm", "grad_norm.png")


def tokenize(batch, tokenizer):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )


def main():
    run_id = get_next_run_id(REPORTS_DIR, prefix="train")
    log_file = REPORTS_DIR / f"train_{run_id}.log"
    plots_dir = REPORTS_DIR / f"train_{run_id}"

    setup_logging(log_file)

    # New model dir per run to preserve previous models
    OUT_DIR = get_next_model_dir(MODELS_DIR, prefix=MODELS_PREFIX)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Starting training run")
    logging.info(f"Run id: {run_id}")
    logging.info(f"Model: {MODEL_NAME}")
    logging.info(f"Data path: {DATA_PATH}")
    logging.info(f"Output dir: {OUT_DIR}")
    logging.info(f"Plots dir: {plots_dir}")

    logging.info("Hyperparameters")
    logging.info(f"SEED={SEED} MAX_LEN={MAX_LEN}")
    logging.info(f"TRAIN_SUBSET_SIZE={TRAIN_SUBSET_SIZE}")
    logging.info(f"NUM_EPOCHS={NUM_EPOCHS}")
    logging.info(f"BATCH_SIZE={BATCH_SIZE}")
    logging.info(f"GRAD_ACCUM={GRAD_ACCUM}")
    logging.info(f"LEARNING_RATE={LEARNING_RATE}")
    logging.info(f"LOGGING_STEPS={LOGGING_STEPS}")
    logging.info(f"SAVE_STEPS={SAVE_STEPS}")
    logging.info(f"SAVE_TOTAL_LIMIT={SAVE_TOTAL_LIMIT}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        "parquet",
        data_files=str(DATA_PATH),
        split="train",
    )

    dataset = dataset.shuffle(seed=SEED).select(range(min(TRAIN_SUBSET_SIZE, len(dataset))))
    logging.info(f"Training samples: {len(dataset)}")

    tokenized = dataset.map(
        lambda x: tokenize(x, tokenizer),
        batched=True,
        remove_columns=["text"],
        desc="Tokenising",
    )

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=str(OUT_DIR),
        overwrite_output_dir=False,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        fp16=torch.cuda.is_available(),
        report_to="none",
        dataloader_num_workers=2,
        logging_strategy="steps",
        save_strategy="steps",
    )

    store = MetricStore()
    metrics_cb = MetricsLoggerCallback(store)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        callbacks=[metrics_cb],
    )

    latest_ckpt = find_latest_checkpoint(OUT_DIR)
    if latest_ckpt is not None:
        logging.info(f"Found checkpoint. Resuming from: {latest_ckpt}")
        trainer.train(resume_from_checkpoint=str(latest_ckpt))
    else:
        logging.info("No checkpoint found. Starting a fresh training run.")
        trainer.train()

    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))
    logging.info(f"Saved fine-tuned model to: {OUT_DIR}")

    save_plots(store, plots_dir)
    logging.info(f"Saved plots to: {plots_dir}")

    logging.info("Training run completed")


if __name__ == "__main__":
    main()

