#!/usr/bin/env python3
"""Four GPU DDP masked-gene pretraining probe for PASA-FM."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from model_transformer import GeneTokenTransformer, GeneTokenSparseTransformer, GeneTokenSparseFoundationTransformer, attention_backend_state


def gather_with_local_grad(x: torch.Tensor, rank: int, world: int) -> torch.Tensor:
    """Gather embeddings across ranks while preserving this rank's gradient path.

    ``dist.all_gather`` does not backpropagate through the gathered tensors on
    all PyTorch versions.  Gathering detached copies and replacing the local
    slice with the original tensor gives every rank the same negative pool and
    keeps gradients for its own positives/queries intact.
    """
    if world == 1:
        return x
    gathered = [torch.zeros_like(x) for _ in range(world)]
    dist.all_gather(gathered, x.detach())
    gathered[rank] = x
    return torch.cat(gathered, dim=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--npz", default="")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad-accum-steps", type=int, default=1,
                    help="number of microbatches per optimizer update")
    ap.add_argument("--seed", type=int, default=20260931)
    ap.add_argument("--train-donors", default="0,1")
    ap.add_argument("--train-assets", default="")
    ap.add_argument("--model-mode", choices=("dense", "sparse", "sparse_foundation", "sparse_covariance", "sparse_contrastive"), default="dense")
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--consistency-weight", type=float, default=1.0)
    ap.add_argument("--variance-weight", type=float, default=8.0)
    ap.add_argument("--covariance-weight", type=float, default=0.2)
    ap.add_argument("--whitening-weight", type=float, default=0.0,
                    help="weight for normalized covariance whitening target")
    ap.add_argument("--contrastive-weight", type=float, default=1.0)
    ap.add_argument("--nonzero-aware-loss", action="store_true",
                    help="upweight masked nonzero targets so reconstruction is not dominated by zeros")
    ap.add_argument("--nonzero-weight", type=float, default=3.0,
                    help="relative weight for masked targets with value > 0 when nonzero-aware loss is enabled")
    ap.add_argument("--cross-rank-negatives", action="store_true",
                    help="use embeddings from all DDP ranks as InfoNCE negatives")
    ap.add_argument("--resolution-token", action="store_true",
                    help="inject audited per-region resolution_um into the global token")
    ap.add_argument("--platform-balanced", action="store_true",
                    help="sample development batches evenly across numeric platform ids 0..6")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
    else:
        rank = 0
        world = 1
        local_rank = 0
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    root = Path(args.root)
    npz_path = Path(args.npz) if args.npz else root / "processed" / "regions_32um.npz"
    z = np.load(npz_path, allow_pickle=False)
    E = z["E"]
    asset = z["asset"]
    donor = z["donor"]
    panels = z["panel"].astype(np.float32)
    raw_platform = np.asarray(z["asset_platform"]).astype(str) if "asset_platform" in z.files else np.full(int(asset.max()) + 1, "7")
    platform_asset = np.full(len(raw_platform), 7, dtype=np.int64)
    for pi, pv in enumerate(raw_platform):
        try:
            q = int(pv)
            if 0 <= q <= 6:
                platform_asset[pi] = q
        except ValueError:
            pass
    if "resolution_um" in z.files:
        raw_resolution = z["resolution_um"].astype(np.float32)
        if len(raw_resolution) == len(E):
            resolutions = raw_resolution
        elif "asset" in z.files and len(raw_resolution) > 0:
            resolutions = raw_resolution[asset]
        else:
            resolutions = np.full(len(E), 8.0, dtype=np.float32)
    else:
        resolutions = np.full(len(E), 8.0, dtype=np.float32)
    train_donors = np.array([int(x) for x in args.train_donors.split(",") if x.strip()], dtype=donor.dtype)
    train_assets = np.array([int(x) for x in args.train_assets.split(",") if x.strip()], dtype=asset.dtype) if args.train_assets else np.array([], dtype=asset.dtype)
    if train_assets.size:
        idx = np.flatnonzero(np.isin(asset, train_assets))
    else:
        idx = np.flatnonzero(np.isin(donor, train_donors))
    idx_by_platform = {p: idx[platform_asset[asset[idx]] == p] for p in range(7)}
    rng = np.random.default_rng(args.seed + rank)
    g = int(E.shape[1])
    model_cls = {"dense": GeneTokenTransformer, "sparse": GeneTokenSparseTransformer,
                 "sparse_foundation": GeneTokenSparseFoundationTransformer,
                 "sparse_covariance": GeneTokenSparseFoundationTransformer,
                 "sparse_contrastive": GeneTokenSparseFoundationTransformer}[args.model_mode]
    model_kwargs = {"d_model": 384, "nhead": 6, "num_layers": args.layers,
                    "dim_ff": 1536, "z_dim": 256}
    if args.model_mode in ("sparse", "sparse_foundation", "sparse_covariance", "sparse_contrastive"):
        model_kwargs["block_size"] = args.block_size
    model_kwargs["use_resolution_token"] = bool(args.resolution_token)
    model = model_cls(g, **model_kwargs).to(device)
    if distributed:
        # Legacy sparse mode leaves z_head/global_encoder unused; sparse_foundation
        # trains both through global reconstruction and the two-view loss.
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    model.train()
    logs = []
    t0 = time.time()
    opt.zero_grad(set_to_none=True)
    for step in range(1, args.steps + 1):
        if args.platform_balanced:
            per = max(1, args.batch // 7)
            ii = np.concatenate([rng.choice(idx_by_platform[p], size=min(per, len(idx_by_platform[p])), replace=False) for p in range(7) if len(idx_by_platform[p]) > 0])
        else:
            ii = rng.choice(idx, size=args.batch, replace=False)
        values = E[ii].astype(np.float32)
        panel = panels[asset[ii]]
        mask = (rng.random(values.shape) < 0.15) & (panel > 0)
        x = values.copy()
        x[mask] = 0.0
        xt = torch.from_numpy(x).to(device)
        observed_panel = panel.copy()
        observed_panel[mask] = 0.0
        pt = torch.from_numpy(observed_panel).to(device)
        rt = torch.from_numpy(resolutions[ii]).to(device)
        target = torch.from_numpy(values).to(device)
        m = torch.from_numpy(mask).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            z1, pred = model(xt, pt, rt if args.resolution_token else None)
            if m.any():
                if args.nonzero_aware_loss:
                    # The public cache is sparse; an unweighted masked MSE can
                    # be minimized by predicting zeros.  Reweight only the
                    # masked nonzero targets, while keeping every masked zero
                    # in the objective.
                    err = (pred[m] - target[m]).square()
                    weights = torch.where(target[m] > 0,
                                          torch.full_like(err, args.nonzero_weight),
                                          torch.ones_like(err))
                    reconstruction_loss = (err * weights).sum() / weights.sum().clamp_min(1.0)
                else:
                    reconstruction_loss = F.mse_loss(pred[m], target[m])
            else:
                reconstruction_loss = pred.square().mean() * 0
            if args.model_mode in ("sparse_foundation", "sparse_covariance", "sparse_contrastive"):
                mask2 = (rng.random(values.shape) < 0.15) & (panel > 0)
                x2 = values.copy(); x2[mask2] = 0.0
                observed_panel2 = panel.copy(); observed_panel2[mask2] = 0.0
                z2, _ = model(torch.from_numpy(x2).to(device), torch.from_numpy(observed_panel2).to(device), rt if args.resolution_token else None)
                consistency_loss = F.mse_loss(z1, z2)
                zcat = torch.cat((z1.float(), z2.float()), dim=0)
                zstd = torch.sqrt(zcat.var(dim=0, unbiased=False) + 1e-4)
                variance_loss = F.relu(0.05 - zstd).mean()
                zc = zcat - zcat.mean(dim=0, keepdim=True)
                cov = (zc.T @ zc) / max(zcat.shape[0] - 1, 1)
                offdiag = cov - torch.diag(torch.diag(cov))
                covariance_loss = offdiag.square().mean()
                whitening_loss = zcat.new_zeros(())
                if args.whitening_weight > 0:
                    # Normalize each latent coordinate before matching its
                    # correlation matrix to identity; this avoids making the
                    # whitening term a scale-explosion objective.
                    zn = zc / torch.sqrt(zcat.var(dim=0, unbiased=False, keepdim=True) + 1e-4)
                    corr = (zn.T @ zn) / max(zcat.shape[0] - 1, 1)
                    whitening_loss = (corr - torch.eye(corr.shape[0], device=corr.device, dtype=corr.dtype)).square().mean()
                if args.model_mode == "sparse_foundation":
                    loss = reconstruction_loss + 0.2 * consistency_loss + 0.5 * variance_loss
                elif args.model_mode == "sparse_contrastive":
                    temperature = 0.1
                    if args.cross_rank_negatives and distributed:
                        z1_pool = gather_with_local_grad(z1.float(), rank, world)
                        z2_pool = gather_with_local_grad(z2.float(), rank, world)
                        labels = torch.arange(z1.shape[0], device=device) + rank * z1.shape[0]
                    else:
                        z1_pool, z2_pool = z1.float(), z2.float()
                        labels = torch.arange(z1.shape[0], device=device)
                    logits12 = (z1.float() @ z2_pool.T) / temperature
                    logits21 = (z2.float() @ z1_pool.T) / temperature
                    contrastive_loss = 0.5 * (F.cross_entropy(logits12, labels) + F.cross_entropy(logits21, labels))
                    loss = (reconstruction_loss + args.contrastive_weight * contrastive_loss
                            + args.variance_weight * variance_loss + args.covariance_weight * covariance_loss
                            + args.whitening_weight * whitening_loss)
                else:
                    loss = (reconstruction_loss + args.consistency_weight * consistency_loss
                            + args.variance_weight * variance_loss + args.covariance_weight * covariance_loss)
            else:
                loss = reconstruction_loss
        scaler.scale(loss / max(args.grad_accum_steps, 1)).backward()
        update_now = (step % max(args.grad_accum_steps, 1) == 0) or (step == args.steps)
        grad_global = 0.0
        grad_z_head = 0.0
        if update_now:
            scaler.unscale_(opt)
        if update_now and args.model_mode in ("sparse_foundation", "sparse_covariance", "sparse_contrastive") and (step == 1 or step % 50 == 0):
            unwrapped = model.module if distributed else model
            grad_global = sum(float(p.grad.float().norm()) for p in unwrapped.global_encoder.parameters() if p.grad is not None)
            grad_z_head = sum(float(p.grad.float().norm()) for p in unwrapped.z_head.parameters() if p.grad is not None)
        if update_now:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        if step == 1 or step % 50 == 0:
            with torch.no_grad():
                zstd_mean = float(zcat.detach().std(dim=0).mean()) if args.model_mode in ("sparse_foundation", "sparse_covariance", "sparse_contrastive") else 0.0
                local = torch.tensor([float(loss.detach()), float(pred.detach().std()), grad_global, grad_z_head, zstd_mean], device=device)
                if distributed:
                    dist.all_reduce(local, op=dist.ReduceOp.SUM)
                    local /= world
                if rank == 0:
                    logs.append({"step": step, "loss": float(local[0]), "pred_std": float(local[1]),
                                 "grad_global": float(local[2]), "grad_z_head": float(local[3]),
                                 "z_std_mean": float(local[4])})
                    print(logs[-1], flush=True)

    if rank == 0:
        wrapped = model.module if distributed else model
        result = {
            "status": "PASS_DDP_SMOKE",
            "layers": args.layers,
            "steps": args.steps,
            "batch_per_gpu": args.batch,
            "grad_accum_steps": args.grad_accum_steps,
            "world_size": world,
            "seed": args.seed,
            "train_donors": [int(x) for x in train_donors.tolist()],
            "train_assets": [int(x) for x in train_assets.tolist()],
            "split_mode": "asset" if train_assets.size else "donor",
            "model_mode": args.model_mode,
            "block_size": args.block_size if args.model_mode in ("sparse", "sparse_foundation", "sparse_covariance", "sparse_contrastive") else None,
            "consistency_weight": args.consistency_weight,
            "variance_weight": args.variance_weight,
            "covariance_weight": args.covariance_weight,
            "whitening_weight": args.whitening_weight,
            "contrastive_weight": args.contrastive_weight,
            "nonzero_aware_loss": bool(args.nonzero_aware_loss),
            "nonzero_weight": args.nonzero_weight,
            "cross_rank_negatives": bool(args.cross_rank_negatives and distributed),
            "resolution_token": bool(args.resolution_token),
            "platform_balanced": bool(args.platform_balanced),
            "observed_panel_includes_random_mask": True,
            "n_train_regions": int(len(idx)),
            "npz": str(npz_path),
            "device": str(device),
            "seconds": time.time() - t0,
            "attention_backend": attention_backend_state(),
            "logs": logs,
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": wrapped.state_dict(), "config": result}, out.with_suffix(".pt"))
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(json.dumps(result, indent=2, ensure_ascii=False))
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
