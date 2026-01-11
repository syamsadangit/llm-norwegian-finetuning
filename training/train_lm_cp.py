# training/train_lm.py
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

MODEL_NAME = "distilgpt2"

# Resolve paths relative to the project root (one level above /training)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data/processed/no_text.parquet"
OUT_DIR = BASE_DIR / "models/distilgpt2-no"


def tokenize(batch, tokenizer):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # GPT-2 style tokenizers typically have no pad token; use EOS for padding
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        "parquet",
        data_files=str(DATA_PATH),
        split="train",
    )

    # Use a small subset for a fast, demonstrative fine-tune
    dataset = dataset.shuffle(seed=42).select(range(min(5000, len(dataset))))

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

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUT_DIR),
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))

    print(f"Saved fine-tuned model to: {OUT_DIR}")


if __name__ == "__main__":
    main()

