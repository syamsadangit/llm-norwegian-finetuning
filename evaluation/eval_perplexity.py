from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config_loader import load_config

CFG = load_config(BASE_DIR / "INPUT")

SEED = int(CFG["SEED"])
MAX_LEN = int(CFG["MAX_LEN"])
VAL_SIZE = int(CFG["VAL_SIZE"])
TRAIN_SUBSET_SIZE = int(CFG["TRAIN_SUBSET_SIZE"])

DATA_PATH = BASE_DIR / str(CFG["DATA_PATH"])
BASE_MODEL_NAME = str(CFG["BASE_MODEL"])

REPORTS_DIR = BASE_DIR / str(CFG.get("REPORTS_DIR", "reports"))
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = REPORTS_DIR / "perplexity.txt"

modified_model_prefix = str(CFG["MODELS_PREFIX"])+str("_1")
FINETUNED_MODEL_DIR = BASE_DIR / str(CFG["MODELS_DIR"]) / modified_model_prefix


def tokenize(batch, tokenizer):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )


@torch.no_grad()
def eval_mean_loss(model, tokenized_dataset, device: torch.device) -> float:
    model.eval()
    losses = []

    for i in range(len(tokenized_dataset)):
        batch = tokenized_dataset[i]
        input_ids = torch.tensor(batch["input_ids"]).unsqueeze(0).to(device)
        attention_mask = torch.tensor(batch["attention_mask"]).unsqueeze(0).to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        losses.append(outputs.loss.item())

    return sum(losses) / len(losses)


def safe_perplexity(mean_loss: float) -> float:
    return math.exp(mean_loss) if mean_loss < 20 else float("inf")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use a single tokenizer for both models to guarantee identical tokenisation.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("parquet", data_files=str(DATA_PATH), split="train")
    ds = ds.shuffle(seed=SEED)

    # Validation slice not used during training (training used indices 0..TRAIN_SUBSET_SIZE-1)
    start = min(TRAIN_SUBSET_SIZE, len(ds))
    end = min(TRAIN_SUBSET_SIZE + VAL_SIZE, len(ds))
    if end <= start:
        raise ValueError(
            f"Not enough data for validation. Need at least {TRAIN_SUBSET_SIZE + 1} rows."
        )

    val = ds.select(range(start, end))
    tok_val = val.map(lambda x: tokenize(x, tokenizer), batched=True, remove_columns=["text"])

    # Evaluate base model
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME).to(device)
    base_loss = eval_mean_loss(base_model, tok_val, device)
    base_ppl = safe_perplexity(base_loss)

    # Evaluate fine-tuned model
    ft_model = AutoModelForCausalLM.from_pretrained(str(FINETUNED_MODEL_DIR)).to(device)
    ft_loss = eval_mean_loss(ft_model, tok_val, device)
    ft_ppl = safe_perplexity(ft_loss)

    delta_loss = ft_loss - base_loss
    delta_ppl = (
        ft_ppl - base_ppl
        if (math.isfinite(ft_ppl) and math.isfinite(base_ppl))
        else float("nan")
    )

    text = (
        "Evaluation: perplexity (base vs fine-tuned)\n"
        f"Dataset: {DATA_PATH}\n"
        f"Validation examples: {len(tok_val)} (indices {start}..{end-1} after shuffle seed={SEED})\n"
        f"Max length: {MAX_LEN}\n\n"
        f"Base model: {BASE_MODEL_NAME}\n"
        f"  Mean loss: {base_loss:.6f}\n"
        f"  Perplexity: {base_ppl:.3f}\n\n"
        f"Fine-tuned model: {FINETUNED_MODEL_DIR}\n"
        f"  Mean loss: {ft_loss:.6f}\n"
        f"  Perplexity: {ft_ppl:.3f}\n\n"
        "Difference (fine-tuned - base)\n"
        f"  Delta loss: {delta_loss:.6f}\n"
        f"  Delta perplexity: {delta_ppl:.3f}\n"
    )

    OUT_FILE.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()

