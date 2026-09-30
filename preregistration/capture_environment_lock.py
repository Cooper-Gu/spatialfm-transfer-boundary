#!/usr/bin/env python3
"""Capture the exact execution-host environment before confirmatory outcomes."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path


def version(module_name: str) -> str | None:
    try:
        return str(importlib.import_module(module_name).__version__)
    except Exception:
        return None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--implementation", type=Path, required=True)
    ap.add_argument("--analysis-contract", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cuda = None
    try:
        torch = importlib.import_module("torch")
        cuda = {
            "version": getattr(torch, "version", None).cuda if getattr(torch, "version", None) else None,
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available() else [],
        }
    except Exception as exc:
        cuda = {"error": repr(exc)}
    result = {
        "study_id": "SFM-TRANSFER-PREREG-001",
        "status": "captured",
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": version("numpy"),
            "scikit-learn": version("sklearn"),
            "torch": version("torch"),
        },
        "cuda": cuda,
        "files": {
            "implementation": {"path": str(args.implementation), "sha256": sha256(args.implementation)},
            "analysis_contract": {"path": str(args.analysis_contract), "sha256": sha256(args.analysis_contract)},
            "protocol": {"path": str(args.protocol), "sha256": sha256(args.protocol)},
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
