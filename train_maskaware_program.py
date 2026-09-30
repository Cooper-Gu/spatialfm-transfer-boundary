#!/usr/bin/env python3
"""Development-only mask-aware local RNA representation candidate.

The candidate receives both the masked expression and an observed-gene mask,
plus a platform one-hot token.  It is deliberately local-only: no spatial
neighbor is allowed to rescue an external cross-platform evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROGRAMS = {
    "immune": ["PTPRC", "CD3D", "CD3E", "CD4", "CD8A", "CD79A", "MS4A1", "LST1", "TYROBP", "FCER1G", "LYZ", "NKG7", "CCL5", "IGKC", "S100A8", "S100A9"],
    "epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "KRT7", "MUC1", "KRT17", "KRT5", "KRT14"],
    "fibroblast": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "COL6A1", "COL6A2", "PDGFRA", "COL5A1", "COL5A2"],
    "vascular": ["PECAM1", "VWF", "KDR", "EMCN", "RAMP2", "ENG", "ESAM", "CLDN5"],
    "proliferation": ["MKI67", "TOP2A", "PCNA", "STMN1", "TYMS", "MCM2", "MCM5"],
    "hypoxia": ["CA9", "VEGFA", "SLC2A1", "LDHA", "HIF1A", "EGLN3", "ADM"],
}


class MaskAware(nn.Module):
    def __init__(self, n: int, p: int, h: int = 512, l: int = 384, platform_dim: int = 8):
        super().__init__()
        self.platform = nn.Sequential(nn.Linear(platform_dim, 32), nn.GELU())
        self.encoder = nn.Sequential(nn.Linear(2 * n + 32, h), nn.LayerNorm(h), nn.GELU(), nn.Linear(h, l))
        self.decoder = nn.Sequential(nn.Linear(l, h), nn.GELU(), nn.Linear(h, n))
        self.local_head = nn.Linear(l, p)

    def forward(self, x: torch.Tensor, observed: torch.Tensor, platform: torch.Tensor):
        z = self.encoder(torch.cat([x, observed, self.platform(platform)], dim=1))
        return z, self.decoder(z), self.local_head(z)


def rank_audit(z: np.ndarray) -> dict:
    a = z - z.mean(0, keepdims=True)
    s = np.linalg.svd(a, compute_uv=False)
    v = s * s
    c = np.cumsum(v) / max(float(v.sum()), 1e-12)
    return {"pr": float(v.sum() ** 2 / max(float((v * v).sum()), 1e-12)), "rank95": int(np.searchsorted(c, 0.95) + 1), "z_std": float(z.std()), "top1": float(v[0] / max(float(v.sum()), 1e-12))}


def platform_ids(values: np.ndarray, assets: np.ndarray) -> np.ndarray:
    raw = np.asarray(values).astype(str)
    out = np.full(len(assets), 7, dtype=np.int64)
    for i, x in enumerate(raw):
        try:
            q = int(x)
            if 0 <= q <= 6:
                out[assets == i] = q
        except ValueError:
            continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--max-train", type=int, default=200000)
    ap.add_argument("--exclude-donors", default="503,504,505,506,507,508,509,510,511,512,513,514,515,516,517,518,519,520,521,522,523")
    ap.add_argument("--program-dropout-rate", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--latent", type=int, default=384)
    ap.add_argument("--platform-balanced", action="store_true")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    z = np.load(a.train_npz, allow_pickle=False)
    E = z["E"].astype("float32")
    asset = z["asset"].astype(np.int64)
    donor = z["donor"].astype(np.int64)
    genes = np.asarray(z["genes"]).astype(str)
    gm = {g.upper(): i for i, g in enumerate(genes)}
    pidx = {k: [gm[g] for g in v if g.upper() in gm] for k, v in PROGRAMS.items()}
    pidx = {k: v for k, v in pidx.items() if len(v) >= 3}
    names = list(pidx)
    excluded = {int(x) for x in a.exclude_donors.split(",") if x.strip()}
    eligible = np.flatnonzero(~np.isin(donor, list(excluded)))
    pool = rng.choice(eligible, size=min(a.max_train, len(eligible)), replace=False)
    panel_asset = z["panel"].astype("float32")
    platform_asset = platform_ids(z.get("asset_platform", np.full(len(np.unique(asset)), "7")), np.arange(len(panel_asset)))
    pool_by_platform = {p: pool[platform_asset[asset[pool]] == p] for p in range(7)}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskAware(E.shape[1], len(names), l=a.latent).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    hist = []
    for step in range(1, a.steps + 1):
        if a.platform_balanced:
            per = max(1, a.batch // 7)
            rows = np.concatenate([rng.choice(pool_by_platform[p], size=min(per, len(pool_by_platform[p])), replace=False) for p in range(7) if len(pool_by_platform[p]) > 0])
        else:
            pos = rng.choice(len(pool), size=min(a.batch, len(pool)), replace=False)
            rows = pool[pos]
        vals = E[rows].copy()
        available = panel_asset[asset[rows]] > 0
        mask = (rng.random(vals.shape) < 0.15) & available
        x = vals.copy()
        if a.program_dropout_rate > 0 and rng.random() < a.program_dropout_rate:
            dn = names[int(rng.integers(len(names)))]
            x[:, pidx[dn]] = 0.0
            mask[:, pidx[dn]] = True
        observed = (available & ~mask).astype("float32")
        pid = platform_asset[asset[rows]]
        ph = np.eye(8, dtype="float32")[pid]
        yt = torch.from_numpy(vals).to(dev)
        xt = torch.from_numpy(x).to(dev)
        ot = torch.from_numpy(observed).to(dev)
        pt = torch.from_numpy(ph).to(dev)
        mt = torch.from_numpy(mask).to(dev)
        cp = torch.stack([yt[:, pidx[n]].mean(1) for n in names], 1)
        zz, rec, prog = model(xt, ot, pt)
        lm = ((rec[mt] - yt[mt]) ** 2).mean() if mt.any() else rec.square().mean() * 0
        lf = ((rec - yt) ** 2).mean()
        lp = torch.nn.functional.mse_loss(prog, cp)
        loss = lm + 0.10 * lf + 0.25 * lp
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 100 == 0 or step == a.steps:
            hist.append({"step": step, "loss": float(loss.detach().cpu()), "masked_loss": float(lm.detach().cpu()), "program_loss": float(lp.detach().cpu()), "z_std": float(zz.detach().std().cpu())})
    with torch.inference_mode():
        ii = pool[: min(4096, len(pool))]
        vals = E[ii]
        available = panel_asset[asset[ii]] > 0
        obs = torch.from_numpy(available.astype("float32")).to(dev)
        ph = torch.from_numpy(np.eye(8, dtype="float32")[platform_asset[asset[ii]]]).to(dev)
        zz, _, _ = model(torch.from_numpy(vals).to(dev), obs, ph)
        audit = rank_audit(zz.cpu().numpy())
    cfg = {"n_genes": int(E.shape[1]), "hidden": 512, "latent": a.latent, "program_names": names, "program_idx": pidx, "platform_dim": 8, "steps": a.steps, "program_dropout_rate": a.program_dropout_rate, "platform_balanced": bool(a.platform_balanced), "max_train": int(len(pool)), "excluded_donors": sorted(excluded), "seed": a.seed, "model_type": "mask_aware_program_ae"}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg, "history": hist, "rank_audit": audit}, out)
    out.with_suffix(".json").write_text(json.dumps({"checkpoint": str(out), "config": cfg, "history": hist, "rank_audit": audit}, indent=2))
    print(json.dumps({"checkpoint": str(out), "history": hist, "rank_audit": audit}, indent=2))


if __name__ == "__main__":
    main()
