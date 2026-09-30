#!/usr/bin/env python3
"""Build a 6,097-gene multi-panel pretraining cache without hiding missing genes.

Each source is mapped by gene name into the fixed master vocabulary. Genes absent
from a source remain zero in E and zero in that source's panel mask; they are not
treated as observed zeros. Asset and donor IDs are made unique across sources.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def mapped_block(path: Path, genes_ref: np.ndarray, asset_offset: int,
                donor_offset: int, platform_override: int | None = None):
    z = np.load(path, allow_pickle=False)
    src_genes = z["genes"].astype(str)
    lookup = {g: i for i, g in enumerate(genes_ref.tolist())}
    src_idx = np.asarray([lookup.get(g, -1) for g in src_genes], dtype=np.int64)
    keep = src_idx >= 0
    E0 = z["E"].astype(np.float16, copy=False)
    E = np.zeros((E0.shape[0], len(genes_ref)), dtype=np.float16)
    E[:, src_idx[keep]] = E0[:, keep]
    A = int(np.max(z["asset"])) + 1
    panel0 = z["panel"].astype(np.uint8, copy=False)
    panel = np.zeros((A, len(genes_ref)), dtype=np.uint8)
    panel[:, src_idx[keep]] = panel0[:, keep]
    asset = z["asset"].astype(np.int32) + asset_offset
    donor0 = z["donor"].astype(np.int32)
    donor_values = np.unique(donor0)
    donor_map = {int(d): donor_offset + i for i, d in enumerate(donor_values.tolist())}
    donor = np.asarray([donor_map[int(d)] for d in donor0], dtype=np.int32)
    asset_donor0 = z["asset_donor"].astype(np.int32)
    asset_donor = np.asarray([donor_map[int(d)] for d in asset_donor0], dtype=np.int32)
    platform = z["asset_platform"].astype(np.int32)
    if platform_override is not None:
        platform = np.full_like(platform, int(platform_override))
    platform = np.clip(platform, 0, 6)
    abbr = z["asset_abbr"].astype(str)
    coords = z["coord"].astype(np.float32, copy=False)
    return {
        "E": E, "coord": coords, "asset": asset, "donor": donor,
        "panel": panel, "asset_abbr": abbr, "asset_donor": asset_donor,
        "asset_platform": platform,
        "n_source_genes": int(len(src_genes)),
        "n_shared_genes": int(keep.sum()),
        "n_regions": int(len(E)), "n_assets": A,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--master", required=True)
    ap.add_argument("--extras", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    master_path = root / args.master
    master = np.load(master_path, allow_pickle=False)
    genes_ref = master["genes"].astype(str)
    blocks = []
    asset_offset = 0
    donor_offset = 0
    # Master block is already in the target vocabulary.
    for path_s, override in [(str(master_path), None)] + [(x, None) for x in args.extras]:
        block = mapped_block(Path(path_s), genes_ref, asset_offset, donor_offset, override)
        blocks.append(block)
        asset_offset += block["n_assets"]
        donor_offset += len(np.unique(block["donor"]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "E": np.concatenate([b["E"] for b in blocks], axis=0),
        "coord": np.concatenate([b["coord"] for b in blocks], axis=0),
        "asset": np.concatenate([b["asset"] for b in blocks], axis=0).astype(np.int16),
        "donor": np.concatenate([b["donor"] for b in blocks], axis=0),
        "panel": np.concatenate([b["panel"] for b in blocks], axis=0),
        "genes": genes_ref,
        "asset_abbr": np.concatenate([b["asset_abbr"] for b in blocks], axis=0),
        "asset_donor": np.concatenate([b["asset_donor"] for b in blocks], axis=0),
        "asset_platform": np.concatenate([b["asset_platform"] for b in blocks], axis=0).astype(np.int16),
    }
    np.savez_compressed(out, **arrays)
    meta = {
        "master": str(master_path),
        "extras": [str(x) for x in args.extras],
        "genes": int(len(genes_ref)),
        "n_regions": int(len(arrays["E"])),
        "n_assets": int(len(arrays["asset_abbr"])),
        "n_donors": int(len(np.unique(arrays["donor"]))),
        "blocks": [{k: b[k] for k in ("n_source_genes", "n_shared_genes", "n_regions", "n_assets")} for b in blocks],
        "missing_genes_are_masked": True,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
