from __future__ import annotations
import json
from pathlib import Path


def _read_jsonl(path):
    path = Path(path)
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def load_registry(registry_path):
    with Path(registry_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_artifact(registry_path, key):
    reg = load_registry(registry_path)
    entry = reg["artifacts"][key]
    if key == "fintrust_registry":
        with Path(entry["path"]).open("r", encoding="utf-8") as f:
            return json.load(f)
    return _read_jsonl(entry["path"])


def load_train(registry_path):
    return load_artifact(registry_path, "train")


def load_validation(registry_path):
    return load_artifact(registry_path, "validation")


def load_finqa_test(registry_path):
    return load_artifact(registry_path, "finqa_test")


def load_fiqa_test(registry_path):
    return load_artifact(registry_path, "fiqa_test")


def load_fintrust_registry(registry_path):
    return load_artifact(registry_path, "fintrust_registry")
