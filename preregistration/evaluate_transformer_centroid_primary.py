#!/usr/bin/env python3
"""Nearest-centroid sensitivity for one frozen Transformer checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_transformer import (GeneTokenSparseFoundationTransformer,
                               GeneTokenSparseTransformer,
                               GeneTokenTransformer)

CLASSES = np.asarray(("L1", "L2", "L3", "L4", "L5", "L6", "WM"))
DONOR_BY_SLICE = {
    **{sid: 0 for sid in ("151507", "151508", "151509", "151510")},
    **{sid: 1 for sid in ("151669", "151670", "151671", "151672")},
    **{sid: 2 for sid in ("151673", "151674", "151675", "151676")},
}


def load_slices(cache_dir: Path):
    out, genes_ref = [], None
    for path in sorted(cache_dir.glob("*.npz")):
        z = np.load(path, allow_pickle=False)
        genes = z["genes"].astype(str)
        if genes_ref is None:
            genes_ref = genes
        elif not np.array_equal(genes_ref, genes):
            raise ValueError(f"gene order mismatch: {path.name}")
        sid = path.stem
        out.append((sid, DONOR_BY_SLICE[sid], z["E"].astype("float32"),
                    z["y"].astype(str), np.broadcast_to(z["panel"].astype("float32"), z["E"].shape).copy()))
    return out, genes_ref


def load_model(checkpoint: Path, n_genes: int, device: torch.device):
    obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = obj.get("config", {})
    mode = cfg.get("model_mode", "dense")
    if mode == "dense":
        cls = GeneTokenTransformer
    elif mode == "sparse":
        cls = GeneTokenSparseTransformer
    elif mode.startswith("sparse"):
        cls = GeneTokenSparseFoundationTransformer
    else:
        raise ValueError(f"Unknown model mode: {mode}")
    kwargs = {"d_model": 384, "nhead": 6, "num_layers": int(cfg.get("layers", 4)),
              "dim_ff": 1536, "z_dim": 256,
              "use_resolution_token": bool(cfg.get("resolution_token", False))}
    if mode != "dense":
        kwargs["block_size"] = int(cfg.get("block_size", 512))
    model = cls(n_genes, **kwargs).to(device)
    model.load_state_dict(obj.get("model", obj.get("state_dict", obj)), strict=True)
    model.eval()
    return model, cfg


def encode(model, E, panel, device, batch_size):
    out = []
    with torch.inference_mode():
        for start in range(0, len(E), batch_size):
            stop = min(start + batch_size, len(E))
            z, _ = model(torch.from_numpy(E[start:stop]).to(device),
                         torch.from_numpy(panel[start:stop]).to(device), None)
            out.append(z.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def centroid_predict(train_x, train_y, test_x):
    mu = train_x.mean(0); sd = train_x.std(0); sd[sd < 1e-6] = 1.0
    A = (train_x - mu) / sd; B = (test_x - mu) / sd
    centroids = np.stack([A[train_y == i].mean(0) for i in range(len(CLASSES))])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True).clip(min=1e-8)
    B /= np.linalg.norm(B, axis=1, keepdims=True).clip(min=1e-8)
    return (B @ centroids.T).argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True); ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256); args = ap.parse_args()
    slices, genes = load_slices(Path(args.cache_dir)); device = torch.device(args.device)
    model, cfg = load_model(Path(args.checkpoint), len(genes), device)
    rows = []
    cmap = {c: i for i, c in enumerate(CLASSES.tolist())}
    for heldout in sorted({s[1] for s in slices}):
        tr = [s for s in slices if s[1] != heldout]; te = [s for s in slices if s[1] == heldout]
        Xtr, Xte = np.concatenate([s[2] for s in tr]), np.concatenate([s[2] for s in te])
        Ptr, Pte = np.concatenate([s[4] for s in tr]), np.concatenate([s[4] for s in te])
        ytr0, yte0 = np.concatenate([s[3] for s in tr]), np.concatenate([s[3] for s in te])
        keep_tr, keep_te = np.isin(ytr0, CLASSES), np.isin(yte0, CLASSES)
        Xtr, Xte, Ptr, Pte = Xtr[keep_tr], Xte[keep_te], Ptr[keep_tr], Pte[keep_te]
        ytr = np.asarray([cmap[x] for x in ytr0[keep_tr]], dtype=np.int64)
        yte = np.asarray([cmap[x] for x in yte0[keep_te]], dtype=np.int64)
        selected = np.logical_and(np.any(Ptr > 0, axis=0), np.any(Pte > 0, axis=0))
        Ztr = encode(model, Xtr * selected, Ptr * selected, device, args.batch_size)
        Zte = encode(model, Xte * selected, Pte * selected, device, args.batch_size)
        for method, A, B in (("model", Ztr, Zte), ("raw", Xtr[:, selected], Xte[:, selected])):
            pred = centroid_predict(A, ytr, B)
            rows.append({"heldout_donor": heldout, "method": method,
                         "macro_f1": float(f1_score(yte, pred, average="macro", labels=np.arange(len(CLASSES)), zero_division=0)),
                         "balanced_accuracy": float(balanced_accuracy_score(yte, pred))})
    out = {"stage": "transformer_centroid_primary", "checkpoint": str(args.checkpoint),
           "config": cfg, "endpoint": "full_ontology", "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({"rows": len(rows), "out": args.out}, ensure_ascii=False))


if __name__ == "__main__":
    main()
