import re
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "data/processed"
REPORT_DIR = BASE_DIR / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_good(text: str, ltol=200, utol=5000, badtol=10) -> bool:
    if text is None:
        return False
    if len(text) < ltol:   # drop very short
        return False
    if len(text) > utol:   # drop very long 
        return False
    # drop lines with too many weird chars
    bad = sum(ch in "{}[]<>|^~" for ch in text)
    if badtol > 10:
        return False
    return True

def main():
    # Norwegian Wikipedia (Bokmaal + Nynorsk) via "wikipedia" dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.no", split="train")

    rows = []
    kept = 0
    total = 0
    dattol=20000

    for ex in tqdm(ds, desc="Cleaning"):
        total += 1
        text = clean_text(ex.get("text", ""))
        if not is_good(text):
            continue

        rows.append({"text": text})
        kept += 1
        if kept >= dattol:
            break

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / "no_text.parquet"
    df.to_parquet(out_path, index=False)

    # Basic stats report
    lengths = df["text"].str.len()
    report = f"""# Data preparation report

Source: wikimedia/wikipedia 20231101.no  
Total scanned: {total}  
Total kept: {len(df)}  

## Length stats (characters)
- mean: {lengths.mean():.1f}
- median: {lengths.median():.1f}
- min: {lengths.min()}
- max: {lengths.max()}

## Example snippet
{df["text"].iloc[0][:500]}...
"""
    (REPORT_DIR / "data_stats.md").write_text(report, encoding="utf-8")
    print(f"Saved dataset -> {out_path}")
    print(f"Saved report  -> {REPORT_DIR / 'data_stats.md'}")

if __name__ == "__main__":
    main()
