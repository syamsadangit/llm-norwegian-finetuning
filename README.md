# Applied Learning Project: Fine-Tuning a Language Model on Norwegian Text

This repository contains a small, learning-focused pipeline for:
- preparing Norwegian text data,
- fine-tuning a pre-trained causal language model (distilgpt2),
- evaluating the fine-tuned model against the base model (loss/perplexity),
- performing prompt-based qualitative comparisons.

This is not a production system; it is an applied learning exercise demonstrating workflow, experiment organisation, and evaluation.

## Project structure
- `data_prep/` – data preparation scripts (Norwegian Wikipedia → cleaned dataset)
- `training/` – fine-tuning script with logging, plots, and run directories
- `evaluation/` – evaluation scripts (perplexity + prompt comparison)
- `reports/` – short result summaries used in the report
- `config_loader.py` – reads key=value settings from `INPUT`

## Setup
Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
Get and prepare Norwegian Wikipedia data from Hugging Face
```bash
python data_prep/prepare_dataset.py
```
## Configuration (INPUT file)

All training and evaluation settings are defined in a single `INPUT` file using simple `key = value` pairs.  
Lines starting with `#` are comments.

```text
SEED = 42
MAX_LEN = 256
DATA_PATH = data/processed/no_text.parquet
BASE_MODEL = distilgpt2
TRAIN_SUBSET_SIZE = 10000
NUM_EPOCHS = 1
BATCH_SIZE = 4
GRAD_ACCUM = 4
LEARNING_RATE = 5e-5
LOGGING_STEPS = 50
SAVE_STEPS = 500
SAVE_TOTAL_LIMIT = 2
VAL_SIZE = 500
MODELS_PREFIX = distilgpt2-no_10k
REPORTS_DIR = reports
MODELS_DIR = models
```

Fine-tune the distilGPT2 model using custom data points defined in `INPUT`
```bash
python training/train_lm.py
```
Evaluate the model using the validation dataset by finding perplexity.
```bash
python evaluation/eval_perplexity.py
```
Qualitative comparison of base mode and fine-tuned model using custom prompts.
```bash
python evaluation/prompt_compare.py
```
## Analysis

All analysis outputs are stored in the `reports/` directory.

- `reports/train_<run_id>.log`  
  Chronological training log containing step-wise loss, learning rate, gradient norm, epoch progress, and total training runtime.  
  Used to inspect training stability and convergence behaviour.

- `reports/train_<run_id>/loss.png`  
  Plot of training loss versus optimisation steps.

- `reports/train_<run_id>/learning_rate.png`  
  Plot of learning rate versus steps.

- `reports/train_<run_id>/grad_norm.png`  
  Plot of gradient norm versus steps.

- `reports/perplexity.txt`  
  Quantitative evaluation summary comparing the base model and the fine-tuned model on held-out data.  
  Reports mean validation loss, perplexity, and their differences.

- `reports/prompt_comparison.md`  
  Qualitative comparison of generated text from the base and fine-tuned models using fixed Norwegian prompts.

Together, these files document training behaviour, quantitative evaluation metrics, and qualitative differences between models.

