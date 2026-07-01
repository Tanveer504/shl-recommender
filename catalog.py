import json
from pathlib import Path

def load_catalog(path: str = None) -> list[dict]:
    if path is None:
        path = Path(__file__).parent / "catalog_clean.json"
    with open(path) as f:
        return json.load(f)
