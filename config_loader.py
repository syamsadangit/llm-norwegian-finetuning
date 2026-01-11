from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def _parse_value(raw: str) -> Any:
    s = raw.strip()

    # Booleans
    if s.lower() in {"true", "false"}:
        return s.lower() == "true"

    # Integers
    try:
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return int(s)
    except Exception:
        pass

    # Floats (incl scientific notation like 5e-5)
    try:
        return float(s)
    except ValueError:
        return s  # string fallback


def load_config(config_path: Path) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    for line in config_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid line (expected key=value): {line}")

        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        cfg[key] = _parse_value(val)

    return cfg


