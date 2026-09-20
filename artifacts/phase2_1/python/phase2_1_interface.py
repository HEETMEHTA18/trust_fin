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


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_registry(registry_path):
    return _read_json(registry_path)


def resolve_artifact(registry_path, key):
    reg = load_registry(registry_path)
    if key not in reg["artifacts"]:
        raise KeyError(f"Unknown artifact {key!r}. Available={sorted(reg['artifacts'])}")
    return reg["artifacts"][key]


def load_artifact(registry_path, key):
    entry = resolve_artifact(registry_path, key)
    path = Path(entry["path"])
    if path.suffix == ".jsonl":
        return _read_jsonl(path)
    if path.suffix == ".json":
        return _read_json(path)
    return path


def load_train(registry_path):
    records = load_artifact(registry_path, "train")
    assert all(not r.get("metadata", {}).get("evaluation_only", False) for r in records)
    return records


def load_validation(registry_path):
    records = load_artifact(registry_path, "validation")
    assert all(not r.get("metadata", {}).get("evaluation_only", False) for r in records)
    return records


def load_finqa_test(registry_path):
    return load_artifact(registry_path, "finqa_test")


def load_fiqa_test(registry_path):
    return load_artifact(registry_path, "fiqa_test")


def load_finben_fomc_test(registry_path):
    return load_artifact(registry_path, "finben_fomc_test")


def load_fintrust_registry(registry_path):
    return load_artifact(registry_path, "fintrust_registry")


def filter_records(records, dataset=None, task=None):
    out = list(records)
    if dataset is not None:
        out = [r for r in out if r.get("dataset") == dataset]
    if task is not None:
        out = [r for r in out if r.get("task") == task]
    return out


def dataset_counts(records):
    counts = {}
    for r in records:
        counts[r["dataset"]] = counts.get(r["dataset"], 0) + 1
    return counts
