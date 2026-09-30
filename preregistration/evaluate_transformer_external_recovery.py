#!/usr/bin/env python3
"""Masked-gene reconstruction on an external 6097-gene cache for a frozen Transformer."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_transformer import (GeneTokenSparseFoundationTransformer,
                               GeneTokenSparseTransformer,
                               GeneTokenTransformer)


def load_model(checkpoint, n_genes, device):
    obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = obj.get("config", {}); mode = cfg.get("model_mode", "dense")
    if mode == "dense":
        cls = GeneTokenTransformer
    elif mode == "sparse":
        cls = GeneTokenSparseTransformer
    elif mode.startswith("sparse"):
        cls = GeneTokenSparseFoundationTransformer
    else:
        raise ValueError(f"Unknown model mode: {mode}")
    kw = {"d_model": 384, "nhead": 6, "num_layers": int(cfg.get("layers", 4)),
          "dim_ff": 1536, "z_dim": 256,
          "use_resolution_token": bool(cfg.get("resolution_token", False))}
    if mode != "dense": kw["block_size"] = int(cfg.get("block_size", 512))
    model = cls(n_genes, **kw).to(device)
    model.load_state_dict(obj.get("model", obj.get("state_dict", obj)), strict=True)
    model.eval(); return model, cfg


def metric(pred, target, baseline):
    mse = float(np.mean((pred - target) ** 2)); d0 = max(float(np.mean(target * target)), 1e-12)
    db = max(float(np.mean((baseline - target) ** 2)), 1e-12)
    corr = float(np.corrcoef(pred, target)[0, 1]) if np.std(pred) > 0 and np.std(target) > 0 else 0.0
    return {"mse": mse, "pearson": corr, "relative_to_zero": float(1 - mse / d0),
            "relative_to_train_mean": float(1 - mse / db)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--external-npz", required=True)
    ap.add_argument("--train-npz", required=True); ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-regions", type=int, default=8192); ap.add_argument("--mask-rate", type=float, default=.15)
    ap.add_argument("--seed", type=int, default=20262001); ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out", required=True); args = ap.parse_args()
    ext = np.load(args.external_npz, allow_pickle=False); tr = np.load(args.train_npz, allow_pickle=False)
    E = ext["E"].astype("float32"); panel = ext["panel"][ext["asset"]].astype(bool)
    rng = np.random.default_rng(args.seed); idx = rng.choice(len(E), size=min(args.n_regions, len(E)), replace=False)
    values, available = E[idx], panel[idx]; mask = (rng.random(values.shape) < args.mask_rate) & available
    x = values.copy(); x[mask] = 0.0; observed = (available & ~mask).astype("float32")
    target = values[mask]; train_mean = tr["E"].astype("float32").mean(axis=0); baseline = np.broadcast_to(train_mean, values.shape)[mask]
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model, cfg = load_model(Path(args.checkpoint), E.shape[1], device); parts = []
    with torch.inference_mode():
        for s in range(0, len(x), args.batch_size):
            e = min(s + args.batch_size, len(x)); _, pred = model(torch.from_numpy(x[s:e]).to(device), torch.from_numpy(observed[s:e]).to(device), None)
            parts.append(pred.float().cpu().numpy())
    pred = np.concatenate(parts, 0)[mask]
    result = {"external_npz": args.external_npz, "train_npz": args.train_npz, "checkpoint": args.checkpoint,
              "seed": int(args.seed), "n_regions": int(len(idx)), "n_masked": int(mask.sum()),
              "available_fraction": float(available.mean()), "baselines": {"zero": metric(np.zeros_like(target), target, baseline),
              "train_panel_mean": metric(baseline, target, baseline)}, "transformer_recovery": metric(pred, target, baseline), "config": cfg}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True); Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False)); print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
