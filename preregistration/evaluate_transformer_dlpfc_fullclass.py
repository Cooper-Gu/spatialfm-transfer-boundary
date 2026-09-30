#!/usr/bin/env python3
"""Evaluate frozen 6097-gene Transformer checkpoints on DLPFC donor holdout.

This is an architecture-extension audit of the fixed full-ontology protocol.
It keeps the MLP endpoint's donor split, class space, probe seeds and baselines,
but changes only the frozen representation checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

# The evaluator lives in ``preregistration`` while the canonical model module
# lives one directory above it on the server.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_transformer import (GeneTokenSparseFoundationTransformer,
                               GeneTokenSparseTransformer,
                               GeneTokenTransformer)

FULL_ONTOLOGY = np.asarray(("L1", "L2", "L3", "L4", "L5", "L6", "WM"))
PANEL_FRACTIONS = (0.10, 0.25, 0.50, 0.75, 1.00)
PROBE_SEEDS = (260, 261, 262)
CLASS_CAP = 1500
DONOR_BY_SLICE = {
    **{sid: 0 for sid in ("151507", "151508", "151509", "151510")},
    **{sid: 1 for sid in ("151669", "151670", "151671", "151672")},
    **{sid: 2 for sid in ("151673", "151674", "151675", "151676")},
}


def balanced_idx(y: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    take = []
    for cls in np.unique(y):
        q = np.flatnonzero(y == cls)
        take.extend(rng.choice(q, min(CLASS_CAP, len(q)), replace=False).tolist())
    out = np.asarray(take, dtype=np.int64)
    rng.shuffle(out)
    return out


def fit_probe(train_x: np.ndarray, train_y: np.ndarray,
              test_x: np.ndarray, seed: int) -> np.ndarray:
    mu = train_x.mean(axis=0)
    sd = train_x.std(axis=0)
    sd[sd < 1e-6] = 1.0
    idx = balanced_idx(train_y, seed)
    clf = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=100, tol=1e-2,
                        average=True, class_weight="balanced",
                        random_state=seed, n_jobs=2)
    clf.fit((train_x[idx] - mu) / sd, train_y[idx])
    return clf.predict((test_x - mu) / sd)


def load_slices(cache_dir: Path):
    out = []
    genes_ref = None
    for path in sorted(cache_dir.glob("*.npz")):
        z = np.load(path, allow_pickle=False)
        genes = z["genes"].astype(str)
        if genes_ref is None:
            genes_ref = genes
        elif not np.array_equal(genes_ref, genes):
            raise ValueError(f"gene order mismatch: {path.name}")
        sid = path.stem
        donor = DONOR_BY_SLICE.get(sid)
        if donor is None:
            raise ValueError(f"missing deterministic donor mapping: {path.name}")
        E = z["E"].astype(np.float32)
        y = z["y"].astype(str)
        p = z["panel"].astype(np.float32)
        if p.shape != (1, len(genes)):
            p = np.ones((1, len(genes)), dtype=np.float32)
        out.append((sid, donor, E, y, np.broadcast_to(p, E.shape).copy()))
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
    kwargs = {"d_model": 384, "nhead": 6,
              "num_layers": int(cfg.get("layers", 4)),
              "dim_ff": 1536, "z_dim": 256,
              "use_resolution_token": bool(cfg.get("resolution_token", False))}
    if mode != "dense":
        kwargs["block_size"] = int(cfg.get("block_size", 512))
    model = cls(n_genes, **kwargs).to(device)
    state = obj.get("model", obj.get("state_dict", obj))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg


def encode(model, E: np.ndarray, panel: np.ndarray, device: torch.device,
           batch_size: int) -> np.ndarray:
    out = []
    with torch.inference_mode():
        for start in range(0, len(E), batch_size):
            stop = min(start + batch_size, len(E))
            x = torch.from_numpy(E[start:stop]).to(device)
            p = torch.from_numpy(panel[start:stop]).to(device)
            z, _ = model(x, p, None)
            out.append(z.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def evaluate(args) -> dict:
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
        Xtr = np.concatenate([s[2] for s in train_s])
        ytr_raw = np.concatenate([s[3] for s in train_s])
        Ptr = np.concatenate([s[4] for s in train_s])
        Xte = np.concatenate([s[2] for s in test_s])
        yte_raw = np.concatenate([s[3] for s in test_s])
        Pte = np.concatenate([s[4] for s in test_s])
        classes = FULL_ONTOLOGY.copy()
        keep_tr = np.isin(ytr_raw, classes)
        keep_te = np.isin(yte_raw, classes)
        Xtr, ytr_raw, Ptr = Xtr[keep_tr], ytr_raw[keep_tr], Ptr[keep_tr]
        Xte, yte_raw, Pte = Xte[keep_te], yte_raw[keep_te], Pte[keep_te]
        cmap = {c: i for i, c in enumerate(classes.tolist())}
        ytr = np.asarray([cmap[c] for c in ytr_raw], dtype=np.int64)
        yte = np.asarray([cmap[c] for c in yte_raw], dtype=np.int64)
        common = np.logical_and(np.any(Ptr > 0, axis=0), np.any(Pte > 0, axis=0))
        order = np.lexsort((genes.astype(str), -np.var(Xtr, axis=0)))
        order = order[common[order]]
        for frac in PANEL_FRACTIONS:
            selected = np.zeros(len(genes), dtype=bool)
            selected[order[:max(1, int(np.ceil(common.sum() * frac)))]] = True
            # The Transformer receives the selected panel as its observed-mask.
            Ztr = encode(model, Xtr * selected, Ptr * selected, device, args.batch_size)
            Zte = encode(model, Xte * selected, Pte * selected, device, args.batch_size)
            pca_n = min(64, int(selected.sum()) - 1)
            pca = PCA(n_components=pca_n, svd_solver="randomized",
                      iterated_power=2, random_state=260 + heldout).fit(Xtr[:, selected])
            methods = {"model": (Ztr, Zte),
                       "raw": (Xtr[:, selected], Xte[:, selected]),
                       "pca": (pca.transform(Xtr[:, selected]), pca.transform(Xte[:, selected]))}
            for probe_seed in PROBE_SEEDS:
                for method, (A, B) in methods.items():
                    pred = fit_probe(A, ytr, B, probe_seed)
                    labels = np.arange(len(classes))
                    cw = f1_score(yte, pred, average=None, labels=labels, zero_division=0)
                    rows.append({
                        "heldout_donor": heldout, "n_training_donors": 2,
                        "endpoint": "full_ontology", "panel_fraction": frac,
                        "n_panel_genes": int(selected.sum()), "probe_seed": probe_seed,
                        "method": method, "macro_f1": float(f1_score(yte, pred, average="macro", labels=labels, zero_division=0)),
                        "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
                        "classwise_f1": {c: float(v) for c, v in zip(classes.tolist(), cw)},
                        "target_class_support": {c: int(np.sum(yte_raw == c)) for c in classes.tolist()},
                        "train_class_support": {c: int(np.sum(ytr_raw == c)) for c in classes.tolist()},
                    })
    return {"stage": "transformer_dlpfc_fullclass", "checkpoint": str(args.checkpoint),
            "config": cfg, "genes": len(genes), "primary_endpoint": "full_ontology",
            "rows": rows, "confirmatory": False}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()
    result = evaluate(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({"rows": len(result["rows"]), "out": args.out}, ensure_ascii=False))


if __name__ == "__main__":
    main()
