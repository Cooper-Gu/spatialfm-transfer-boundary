#!/usr/bin/env python3
"""Prospectively frozen SFM-TRANSFER-PREREG-001 evaluator.

The real run consumes one .npz cache per spatial slice.  The cache must contain
E (spots x common genes), genes, y and optionally panel, P/platform_onehot and
donor_id.  Panel selection is fit from training-donor variance only; target
labels are used only by the fixed probe after the representation is created.

`--smoke-test` is a read-only implementation check.  It uses deterministic
synthetic data and a synthetic checkpoint with the same loader contract, and
must never be reported as a confirmatory outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform as py_platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch import nn


STUDY_ID = "SFM-TRANSFER-PREREG-001"
PANEL_FRACTIONS = (0.10, 0.25, 0.50, 0.75, 1.00)
TRAINING_DONOR_COUNTS = (1, 2)
PROBE_SEEDS = (260, 261, 262)
CLASS_CAP = 1500
PLATFORM_DIM = 8
FULL_ONTOLOGY = np.asarray(("L1", "L2", "L3", "L4", "L5", "L6", "WM"))
SPATIALLIBD_DONOR_BY_SAMPLE = {
    **{sid: 0 for sid in ("151507", "151508", "151509", "151510")},
    **{sid: 1 for sid in ("151669", "151670", "151671", "151672")},
    **{sid: 2 for sid in ("151673", "151674", "151675", "151676")},
}


class FrozenRepresentation(nn.Module):
    """Loader-compatible label_heldout MLP used by the existing checkpoint family."""

    def __init__(self, n_genes: int, n_programs: int, hidden: int, latent: int,
                 platform_dim: int = PLATFORM_DIM) -> None:
        super().__init__()
        self.platform = nn.Sequential(nn.Linear(platform_dim, 32), nn.GELU())
        self.encoder = nn.Sequential(
            nn.Linear(2 * n_genes + 32, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, latent),
        )

    def forward(self, x: torch.Tensor, observed: torch.Tensor,
                platform: torch.Tensor) -> torch.Tensor:
        return self.encoder(torch.cat([x, observed, self.platform(platform)], dim=1))


@dataclass
class Slice:
    name: str
    donor: int
    E: np.ndarray
    y: np.ndarray
    genes: np.ndarray
    panel: np.ndarray
    platform: np.ndarray


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def balanced_idx(y: np.ndarray, max_per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    take: list[int] = []
    for cls in np.unique(y):
        q = np.flatnonzero(y == cls)
        take.extend(rng.choice(q, min(max_per_class, len(q)), replace=False).tolist())
    out = np.asarray(take, dtype=np.int64)
    rng.shuffle(out)
    return out


def fit_probe(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray,
              seed: int) -> tuple[np.ndarray, int]:
    mu = train_x.mean(axis=0)
    sd = train_x.std(axis=0)
    sd[sd < 1e-6] = 1.0
    idx = balanced_idx(train_y, CLASS_CAP, seed)
    clf = SGDClassifier(
        loss="log_loss", alpha=1e-4, max_iter=100, tol=1e-2,
        average=True, class_weight="balanced", random_state=seed, n_jobs=2,
    )
    clf.fit((train_x[idx] - mu) / sd, train_y[idx])
    return clf.predict((test_x - mu) / sd), int(len(idx))


def load_checkpoint(path: Path, n_genes: int, device: torch.device) -> nn.Module:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    cfg = obj.get("config", {})
    program_names = cfg.get("program_names", [])
    model = FrozenRepresentation(
        n_genes=n_genes,
        n_programs=len(program_names),
        hidden=int(cfg.get("hidden", 128)),
        latent=int(cfg.get("latent", 64)),
        platform_dim=int(cfg.get("platform_dim", PLATFORM_DIM)),
    ).to(device)
    state = obj.get("model", obj.get("state_dict", obj))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def encode(model: nn.Module, E: np.ndarray, observed: np.ndarray,
           platform: np.ndarray, selected: np.ndarray,
           device: torch.device) -> np.ndarray:
    # Keep the checkpoint input dimension fixed.  Unselected genes are masked,
    # and the observed mask is passed as the second input block.
    x = E.astype("float32", copy=True)
    x[:, ~selected] = 0.0
    o = np.broadcast_to(selected.astype("float32"), x.shape).copy()
    out: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(x), 512):
            xx = torch.from_numpy(x[start:start + 512]).to(device)
            oo = torch.from_numpy(o[start:start + 512]).to(device)
            pp = torch.from_numpy(platform[start:start + 512]).to(device)
            out.append(model(xx, oo, pp).cpu().numpy())
    return np.concatenate(out, axis=0)


def panel_from_training(slices: Iterable[Slice], genes: np.ndarray,
                        common_mask: np.ndarray, fraction: float) -> np.ndarray:
    E = np.concatenate([s.E for s in slices], axis=0)
    # Variance is the only quantity used for panel selection.  No labels,
    # target donor values or test-slice metadata enter this function.
    variance = np.nanvar(E, axis=0)
    variance[~common_mask] = -np.inf
    order = np.lexsort((genes.astype(str), -variance))
    n = max(1, int(np.ceil(int(common_mask.sum()) * fraction)))
    return np.isin(np.arange(len(genes)), order[:n])


def load_slices(cache_dir: Path, donor_map: dict[str, int] | None,
                max_spots: int | None = None) -> list[Slice]:
    files = sorted(cache_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz slices found in {cache_dir}")
    out: list[Slice] = []
    genes_ref: np.ndarray | None = None
    for path in files:
        z = np.load(path, allow_pickle=False)
        genes = z["genes"].astype(str)
        if genes_ref is None:
            genes_ref = genes
        elif not np.array_equal(genes_ref, genes):
            raise ValueError(f"gene order mismatch: {path.name}")
        key = path.stem
        donor: Any = SPATIALLIBD_DONOR_BY_SAMPLE.get(key)
        if donor is None and "donor_id" in z:
            donor = z["donor_id"].item()
        if donor is None and "donor" in z:
            donor_values = np.unique(z["donor"].reshape(-1))
            if len(donor_values) == 1:
                donor = int(donor_values[0])
        if donor is None and "asset_donor" in z:
            donor = int(z["asset_donor"].reshape(-1)[0])
        if donor is None and donor_map is not None:
            donor = donor_map.get(key)
        if donor is None:
            raise ValueError(f"missing deterministic donor mapping for {path.name}")
        E = z["E"].astype("float32")
        y = z["y"].astype(str)
        if len(E) != len(y):
            raise ValueError(f"E/y length mismatch: {path.name}")
        if max_spots is not None:
            E, y = E[:max_spots], y[:max_spots]
        panel = z["panel"].astype("float32").reshape(-1) if "panel" in z else np.ones(len(genes), dtype="float32")
        if panel.ndim != 1 or len(panel) != len(genes):
            raise ValueError(f"panel must be one-dimensional and gene-aligned: {path.name}")
        if "platform_onehot" in z:
            p = z["platform_onehot"].astype("float32")
            p = np.broadcast_to(p, (len(E), len(p))).copy() if p.ndim == 1 else p[:len(E)]
        elif "P" in z:
            p = z["P"].astype("float32")[:len(E)]
        else:
            p = np.zeros((len(E), PLATFORM_DIM), dtype="float32")
            p[:, 0] = 1.0
        out.append(Slice(key, int(donor), E, y, genes, panel, p))
    return out


def _endpoint_arrays(ytr_raw: np.ndarray, yte_raw: np.ndarray,
                     endpoint: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fixed class space and endpoint-specific train/test masks.

    The original preregistration used only the train/test class intersection.
    This reviewer repair keeps that result as a labelled sensitivity analysis,
    while the primary endpoint scores the full cortical ontology.  Classes
    absent from the training pool remain in the fixed label space and receive
    zero F1 when they have target support; this is deterministic and avoids
    silently dropping a donor-specific biological class.
    """
    if endpoint == "full_ontology":
        classes = FULL_ONTOLOGY.copy()
        keep_tr = np.isin(ytr_raw, classes)
        keep_te = np.isin(yte_raw, classes)
    elif endpoint == "target_present":
        classes = np.asarray(sorted(set(yte_raw)), dtype=str)
        keep_tr = np.isin(ytr_raw, classes)
        keep_te = np.isin(yte_raw, classes)
    elif endpoint == "shared_class_sensitivity":
        classes = np.asarray(sorted(set(ytr_raw) & set(yte_raw)), dtype=str)
        keep_tr = np.isin(ytr_raw, classes)
        keep_te = np.isin(yte_raw, classes)
    else:
        raise ValueError(f"unknown endpoint: {endpoint}")
    if len(classes) < 2:
        raise ValueError(f"{endpoint}: fewer than two target classes")
    return classes, keep_tr, keep_te


def evaluate(slices: list[Slice], model: nn.Module, device: torch.device,
             checkpoint: str, stage: str, common_mask: np.ndarray) -> dict[str, Any]:
    genes = slices[0].genes
    donors = sorted({s.donor for s in slices})
    if len(donors) != 3:
        raise ValueError(f"protocol requires exactly 3 donors, got {donors}")
    rows: list[dict[str, Any]] = []
    for heldout in donors:
        train_pool = [s for s in slices if s.donor != heldout]
        test_pool = [s for s in slices if s.donor == heldout]
        for n_train_donors in TRAINING_DONOR_COUNTS:
            train_donors = sorted({s.donor for s in train_pool})
            if n_train_donors == 1:
                train_donors = train_donors[:1]  # deterministic lower donor ID
            train_slices = [s for s in train_pool if s.donor in train_donors]
            Xtr = np.concatenate([s.E for s in train_slices], axis=0)
            ytr_raw = np.concatenate([s.y for s in train_slices], axis=0)
            Ptr = np.concatenate([s.platform for s in train_slices], axis=0)
            Xte = np.concatenate([s.E for s in test_pool], axis=0)
            yte_raw = np.concatenate([s.y for s in test_pool], axis=0)
            Pte = np.concatenate([s.platform for s in test_pool], axis=0)
            for endpoint in ("full_ontology", "target_present", "shared_class_sensitivity"):
                classes, keep_tr, keep_te = _endpoint_arrays(ytr_raw, yte_raw, endpoint)
                Xtr_e, ytr_e, Ptr_e = Xtr[keep_tr], ytr_raw[keep_tr], Ptr[keep_tr]
                Xte_e, yte_e, Pte_e = Xte[keep_te], yte_raw[keep_te], Pte[keep_te]
                cmap = {c: i for i, c in enumerate(classes.tolist())}
                ytr = np.asarray([cmap[c] for c in ytr_e], dtype=np.int64)
                yte = np.asarray([cmap[c] for c in yte_e], dtype=np.int64)
                class_support_target = {c: int(np.sum(yte_e == c)) for c in classes.tolist()}
                class_support_train = {c: int(np.sum(ytr_e == c)) for c in classes.tolist()}
                for fraction in PANEL_FRACTIONS:
                    selected = panel_from_training(train_slices, genes, common_mask, fraction)
                    Ztr = encode(model, Xtr_e, np.broadcast_to(selected, Xtr_e.shape), Ptr_e, selected, device)
                    Zte = encode(model, Xte_e, np.broadcast_to(selected, Xte_e.shape), Pte_e, selected, device)
                    pca_n = min(64, int(selected.sum()) - 1)
                    pca = None
                    if pca_n >= 1:
                        pca = PCA(n_components=pca_n, svd_solver="randomized",
                                  iterated_power=2, random_state=260 + heldout).fit(Xtr_e[:, selected])
                    for probe_seed in PROBE_SEEDS:
                        methods: dict[str, np.ndarray] = {
                            "model": Ztr,
                            "raw": Xtr_e[:, selected],
                        }
                        if pca is not None:
                            methods["pca"] = pca.transform(Xtr_e[:, selected])
                        for method, A in methods.items():
                            B = {"model": Zte, "raw": Xte_e[:, selected],
                                 "pca": pca.transform(Xte_e[:, selected]) if pca is not None else None}[method]
                            if B is None:
                                continue
                            pred, n_fit = fit_probe(A, ytr, B, probe_seed)
                            labels = np.arange(len(classes))
                            classwise = f1_score(yte, pred, average=None, labels=labels, zero_division=0)
                            rows.append({
                                "heldout_donor": heldout,
                                "training_donors": train_donors,
                                "n_training_donors": n_train_donors,
                                "endpoint": endpoint,
                                "panel_fraction": fraction,
                                "n_panel_genes": int(selected.sum()),
                                "probe_seed": probe_seed,
                                "method": method,
                                "macro_f1": float(f1_score(yte, pred, average="macro",
                                                           labels=labels, zero_division=0)),
                                "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
                                "classwise_f1": {c: float(v) for c, v in zip(classes.tolist(), classwise)},
                                "n_train_probe": n_fit,
                                "n_train": int(len(ytr)),
                                "n_test": int(len(yte)),
                                "target_class_support": class_support_target,
                                "train_class_support": class_support_train,
                                "classes": classes.tolist(),
                            })
    return {
        "study_id": STUDY_ID,
        "stage": stage,
        "source": "spatialLIBD Human DLPFC Visium",
        "holdout": "leave-one-donor-out",
        "common_genes": int(common_mask.sum()),
        "model_input_genes": int(len(genes)),
        "checkpoint": checkpoint,
        "panel_fractions": list(PANEL_FRACTIONS),
        "training_donor_counts": list(TRAINING_DONOR_COUNTS),
        "probe_seeds": list(PROBE_SEEDS),
        "primary_endpoint": "full_ontology",
        "sensitivity_endpoints": ["target_present", "shared_class_sensitivity"],
        "full_ontology": FULL_ONTOLOGY.tolist(),
        "scoring_rule": "fixed L1/L2/L3/L4/L5/L6/WM label space; absent target support is reported as zero-support and never silently dropped; absent training class yields zero F1 when target support exists",
        "rows": rows,
        "target_labels_used_for_panel_selection": False,
        "confirmatory": False if stage in {"smoke", "real_smoke"} else True,
    }


def smoke_test() -> dict[str, Any]:
    rng = np.random.default_rng(20260928)
    n_genes, hidden, latent = 40, 32, 16
    model = FrozenRepresentation(n_genes, 4, hidden, latent)
    model.eval()
    slices: list[Slice] = []
    genes = np.asarray([f"G{i:03d}" for i in range(n_genes)])
    for donor in range(3):
        for section in range(2):
            n = 36
            y = np.asarray([f"L{i % 3}" for i in range(n)])
            E = rng.normal(size=(n, n_genes)).astype("float32")
            E[:, 0] += np.asarray([0.0, 1.0, 2.0] * 12, dtype="float32")
            P = np.zeros((n, PLATFORM_DIM), dtype="float32"); P[:, 0] = 1.0
            slices.append(Slice(f"synthetic_d{donor}_s{section}", donor, E, y, genes,
                                np.ones(n_genes, dtype="float32"), P))
    common_mask = np.ones(n_genes, dtype=bool)
    out = evaluate(slices, model, torch.device("cpu"), "synthetic_smoke_checkpoint", "smoke", common_mask)
    if len(out["rows"]) != 3 * 2 * 3 * 5 * 3 * 3:
        raise AssertionError(f"unexpected row count: {len(out['rows'])}")
    out["smoke_pass"] = True
    out["note"] = "Synthetic implementation smoke only; no confirmatory result."
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--donor-map-json", type=Path)
    ap.add_argument("--max-spots-per-slice", type=int)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--stage", default="confirmatory")
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()
    if args.smoke_test:
        result = smoke_test()
    else:
        if args.cache_dir is None or args.checkpoint is None:
            ap.error("real evaluation requires --cache-dir and --checkpoint")
        donor_map = None
        if args.donor_map_json:
            donor_map = {str(k): int(v) for k, v in json.loads(args.donor_map_json.read_text()).items()}
        slices = load_slices(args.cache_dir, donor_map, args.max_spots_per_slice)
        device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        model = load_checkpoint(args.checkpoint, len(slices[0].genes), device)
        common_masks = [s.panel.astype(bool) for s in slices]
        common_mask = np.logical_and.reduce(common_masks)
        if int(common_mask.sum()) != 5442:
            raise ValueError(f"protocol requires 5,442 common genes, observed {int(common_mask.sum())}")
        result = evaluate(slices, model, device, str(args.checkpoint), args.stage, common_mask)
    result["python"] = py_platform.python_version()
    result["torch"] = torch.__version__
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({k: result[k] for k in result if k not in {"rows"}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
