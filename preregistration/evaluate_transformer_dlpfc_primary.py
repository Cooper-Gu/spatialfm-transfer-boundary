#!/usr/bin/env python3
"""Fast primary full-panel DLPFC endpoint for a frozen Transformer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score

from evaluate_transformer_dlpfc_fullclass import (FULL_ONTOLOGY, fit_probe,
                                                      load_model, load_slices)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    slices, genes = load_slices(Path(args.cache_dir))
    if len(genes) != 6097:
        raise ValueError(f"expected 6097 genes, got {len(genes)}")
    donors = sorted({s[1] for s in slices})
    device = torch.device(args.device)
    model, cfg = load_model(Path(args.checkpoint), len(genes), device)
    rows = []
    for heldout in donors:
        train_s = [s for s in slices if s[1] != heldout]
        test_s = [s for s in slices if s[1] == heldout]
        Xtr = np.concatenate([s[2] for s in train_s]); ytr_raw = np.concatenate([s[3] for s in train_s]); Ptr = np.concatenate([s[4] for s in train_s])
        Xte = np.concatenate([s[2] for s in test_s]); yte_raw = np.concatenate([s[3] for s in test_s]); Pte = np.concatenate([s[4] for s in test_s])
        keep_tr = np.isin(ytr_raw, FULL_ONTOLOGY); keep_te = np.isin(yte_raw, FULL_ONTOLOGY)
        Xtr, ytr_raw, Ptr = Xtr[keep_tr], ytr_raw[keep_tr], Ptr[keep_tr]
        Xte, yte_raw, Pte = Xte[keep_te], yte_raw[keep_te], Pte[keep_te]
        cmap = {c: i for i, c in enumerate(FULL_ONTOLOGY.tolist())}
        ytr = np.asarray([cmap[c] for c in ytr_raw], dtype=np.int64)
        yte = np.asarray([cmap[c] for c in yte_raw], dtype=np.int64)
        selected = np.ones(len(genes), dtype=bool)
        def encode(E, P):
            out = []
            with torch.inference_mode():
                for s in range(0, len(E), args.batch_size):
                    x = torch.from_numpy(E[s:s + args.batch_size]).to(device)
                    p = torch.from_numpy(P[s:s + args.batch_size]).to(device)
                    z, _ = model(x, p, None)
                    out.append(z.float().cpu().numpy())
            return np.concatenate(out, axis=0)
        Ztr, Zte = encode(Xtr, Ptr), encode(Xte, Pte)
        mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0); sd[sd < 1e-6] = 1.0
        methods = {"model": (Ztr, Zte), "raw": ((Xtr - mu) / sd, (Xte - mu) / sd)}
        for seed in (260, 261, 262):
            for method, (A, B) in methods.items():
                pred = fit_probe(A, ytr, B, seed)
                rows.append({"heldout_donor": heldout, "endpoint": "full_ontology", "panel_fraction": 1.0,
                             "probe_seed": seed, "method": method,
                             "macro_f1": float(f1_score(yte, pred, average="macro", labels=np.arange(7), zero_division=0)),
                             "balanced_accuracy": float(balanced_accuracy_score(yte, pred))})
    result = {"stage": "transformer_dlpfc_primary_fullpanel", "checkpoint": args.checkpoint,
              "config": cfg, "genes": len(genes), "rows": rows}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False)); print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
