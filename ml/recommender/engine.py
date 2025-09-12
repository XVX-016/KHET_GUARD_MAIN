import json
from pathlib import Path
from typing import Dict, Any

DATA: Dict[str, Any] = {}

def _load() -> None:
    global DATA
    path = Path(__file__).parent / "pesticide_map.json"
    with open(path, "r", encoding="utf-8") as f:
        DATA = json.load(f)


def recommend(pest_class: str) -> Dict[str, Any]:
    if not DATA:
        _load()
    return DATA.get(pest_class.lower(), {"recommended": [], "dosage": "N/A", "safety": "Consult expert"})


