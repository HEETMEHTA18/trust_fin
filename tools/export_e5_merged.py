#!/usr/bin/env python3
"""Export the frozen E5 LoRA adapter as a standalone merged model.

This writes a *copy*; frozen artifacts under artifacts/phase3/ are only read.
The merged model is numerically intended to match base + adapter, but it is a
new derived artifact and must not be described as the original checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADAPTER = ROOT / "artifacts/phase3/checkpoints/E5"
DEFAULT_OUT = ROOT / "deliverables/export/TrustFin-E5-merged"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    adapter = Path(args.adapter).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    cfg_path = adapter / "adapter_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"adapter_config.json missing at {adapter}")
    weights = [adapter / n for n in ("adapter_model.safetensors", "adapter_model.bin")]
    weights = [p for p in weights if p.is_file()]
    if not weights:
        raise FileNotFoundError(f"adapter weights missing at {adapter}")

    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out} already exists; pass --overwrite to replace it")
        shutil.rmtree(out)

    conf = json.loads(cfg_path.read_text())
    base_id = args.base_model or conf["base_model_name_or_path"]
    dtype = getattr(torch, args.dtype)
    print(f"[export] base={base_id} adapter={adapter} dtype={args.dtype} device={args.device}")

    tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_id,
        trust_remote_code=False,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model = model.merge_and_unload()
    model.eval()
    if args.device == "cuda":
        model = model.to("cuda")

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True, max_shard_size="4GB")
    tok.save_pretrained(out)

    # Preserve the exact chat template used at training/evaluation time.
    tpl = adapter / "chat_template.jinja"
    if tpl.is_file():
        tok.chat_template = tpl.read_text()
        (out / "chat_template.jinja").write_text(tok.chat_template)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "merged_derived_model",
        "note": "Copy of Qwen/Qwen3-1.7B with E5 LoRA merged. Derived artifact, not the frozen checkpoint.",
        "base_model": base_id,
        "adapter": str(adapter),
        "adapter_config_sha256": sha256(cfg_path),
        "adapter_weights": {p.name: sha256(p) for p in weights},
        "merge_method": "peft.PeftModel.merge_and_unload",
        "dtype": args.dtype,
        "versions": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "peft": __import__("peft").__version__,
        },
        "files": {},
    }

    # Smoke test: the merged model should recover the documented E5 program.
    smoke = {}
    if not args.skip_smoke:
        messages = [
            {"role": "system", "content": "You are a financial reasoning assistant. Use only the supplied financial evidence. Return a valid FinQA program and the final answer."},
            {"role": "user", "content": "<FINQA>\nSolve the financial question from the supplied evidence.\n\nFinancial text:\nRevenue was 800 million in 2024 and 1000 million in 2025.\n\nFinancial table:\n\n\nQuestion:\nWhat was the increase in revenue, in millions?"},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        batch = tok(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            gen = model.generate(**batch, max_new_tokens=64, do_sample=False, pad_token_id=tok.pad_token_id)
        smoke = {
            "prompt": "Revenue 800->1000; increase?",
            "raw_output": tok.decode(gen[0, batch["input_ids"].shape[1]:], skip_special_tokens=True),
        }
        print("[export] smoke output:", smoke["raw_output"].replace("\n", " | "))
    manifest["smoke_test"] = smoke

    for p in sorted(out.iterdir()):
        if p.is_file() and p.name != "export_manifest.json":
            manifest["files"][p.name] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    (out / "export_manifest.json").write_text(json.dumps(manifest, indent=2))

    total = sum(v["bytes"] for v in manifest["files"].values())
    print(f"[export] wrote {len(manifest['files'])} files, {total/1e9:.2f} GB -> {out}")


if __name__ == "__main__":
    main()
