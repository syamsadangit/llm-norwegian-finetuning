from __future__ import annotations

from pathlib import Path
import datetime
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config_loader import load_config

CFG = load_config(BASE_DIR / "INPUT")

REPORTS_DIR = BASE_DIR / str(CFG.get("REPORTS_DIR", "reports"))
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = REPORTS_DIR / "prompt_comparison.md"

BASE_MODEL = str(CFG["BASE_MODEL"])

modified_model_prefix = str(CFG["MODELS_PREFIX"])+str("_1")
FINETUNED_MODEL_DIR = BASE_DIR / str(CFG["MODELS_DIR"]) / modified_model_prefix

MAX_NEW_TOKENS = int(CFG.get("MAX_NEW_TOKENS", 60))
TEMPERATURE = float(CFG.get("TEMPERATURE", 0.8))

PROMPTS = [
    "Norge er kjent for",
    "Mo i Rana er en by som",
    "Et bibliotek har som oppgave å",
    "Kunstig intelligens kan brukes til å",
    "En kort forklaring av språkmodeller:",
    "Samiske språk er viktige fordi",
    "Fordeler og ulemper med maskinlæring er",
    "I en forskningsrapport bør man",
    "Datasettkvalitet er viktig fordi",
    "For å evaluere en språkmodell kan man",
]


def load(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return tok, model


@torch.no_grad()
def generate(tok, model, prompt: str, max_new_tokens: int, temperature: float):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(outputs[0], skip_special_tokens=True)


def main():
    base_tok, base_model = load(BASE_MODEL)
    ft_tok, ft_model = load(str(FINETUNED_MODEL_DIR))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model.to(device)
    ft_model.to(device)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = []
    lines.append("# Prompt comparison (base vs fine-tuned)\n\n")
    lines.append(f"- Date: {now}\n")
    lines.append(f"- Base model: {BASE_MODEL}\n")
    lines.append(f"- Fine-tuned model: {FINETUNED_MODEL_DIR}\n")
    lines.append(f"- max_new_tokens: {MAX_NEW_TOKENS}\n")
    lines.append(f"- temperature: {TEMPERATURE}\n\n")

    for p in PROMPTS:
        lines.append(f"## Prompt: {p}\n\n")

        base_out = generate(base_tok, base_model, p, MAX_NEW_TOKENS, TEMPERATURE)
        ft_out = generate(ft_tok, ft_model, p, MAX_NEW_TOKENS, TEMPERATURE)

        lines.append("### Base output\n")
        lines.append("```text\n" + base_out + "\n```\n\n")

        lines.append("### Fine-tuned output\n")
        lines.append("```text\n" + ft_out + "\n```\n\n")

    OUT_FILE.write_text("".join(lines), encoding="utf-8")
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()

