import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


def load_json_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(p, 'r') as f:
        return json.load(f)


def save_json_config(path: str, data: Dict[str, Any]):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w') as f:
        json.dump(data, f, indent=2)


