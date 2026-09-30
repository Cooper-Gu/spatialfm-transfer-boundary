#!/usr/bin/env python
"""Region-level gene-token Transformer.

The model consumes a fixed gene vocabulary plus an assay/panel mask.  It is
deliberately small enough for a smoke test, while retaining the same interface
for the planned full pretraining run.
"""
from __future__ import annotations

import torch
from torch import nn


class GeneTokenTransformer(nn.Module):
    def __init__(self, n_genes: int, d_model: int = 384, nhead: int = 6,
                 num_layers: int = 2, dim_ff: int = 1536,
                 z_dim: int = 256, dropout: float = 0.1,
                 use_resolution_token: bool = False):
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.n_genes = n_genes
        self.d_model = d_model
        self.use_resolution_token = bool(use_resolution_token)
        self.gene_emb = nn.Embedding(n_genes, d_model)
        self.value_proj = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.mask_emb = nn.Embedding(2, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                             norm=nn.LayerNorm(d_model))
        self.z_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, z_dim))
        self.value_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        if self.use_resolution_token:
            self.resolution_proj = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self, values: torch.Tensor, panel: torch.Tensor, resolution_um: torch.Tensor | None = None):
        # values/panel: [batch, genes]. panel is 1 for assayed genes.
        b, g = values.shape
        if g != self.n_genes:
            raise ValueError(f"expected {self.n_genes} genes, got {g}")
        ids = torch.arange(g, device=values.device).unsqueeze(0).expand(b, -1)
        observed = (panel > 0).long()
        tok = self.gene_emb(ids)
        tok = tok + self.value_proj(values.unsqueeze(-1)) + self.mask_emb(observed)
        cls = self.cls.expand(b, -1, -1)
        if self.use_resolution_token:
            if resolution_um is None:
                raise ValueError("resolution_um is required when use_resolution_token=True")
            r = torch.log2(resolution_um.float().view(b, 1, 1).clamp_min(1.0) / 8.0)
            cls = cls + self.resolution_proj(r)
        h = self.encoder(torch.cat([cls, tok], dim=1))
        z = torch.nn.functional.normalize(self.z_head(h[:, 0]), dim=-1)
        pred = self.value_head(h[:, 1:]).squeeze(-1)
        return z, pred


class GeneTokenSparseTransformer(nn.Module):
    """Block sparse gene-token Transformer for vocabularies larger than 2k.

    Gene tokens attend within fixed blocks. A compact global encoder attends to
    one summary token per block, so memory grows linearly in the number of
    blocks instead of quadratically in the full gene vocabulary.
    """
    def __init__(self, n_genes: int, d_model: int = 384, nhead: int = 6,
                 num_layers: int = 4, dim_ff: int = 1536,
                 z_dim: int = 256, dropout: float = 0.1,
                 block_size: int = 512,
                 use_resolution_token: bool = False):
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.n_genes = n_genes
        self.block_size = block_size
        self.d_model = d_model
        self.use_resolution_token = bool(use_resolution_token)
        self.gene_emb = nn.Embedding(n_genes, d_model)
        self.value_proj = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.mask_emb = nn.Embedding(2, d_model)
        self.block_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.global_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.block_cls, std=0.02)
        nn.init.normal_(self.global_cls, std=0.02)
        local_layers = max(1, num_layers - 1)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.local_encoder = nn.TransformerEncoder(layer, num_layers=local_layers,
                                                    norm=nn.LayerNorm(d_model))
        global_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.global_encoder = nn.TransformerEncoder(global_layer, num_layers=1,
                                                     norm=nn.LayerNorm(d_model))
        self.z_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, z_dim))
        self.value_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        if self.use_resolution_token:
            self.resolution_proj = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self, values: torch.Tensor, panel: torch.Tensor, resolution_um: torch.Tensor | None = None):
        b, g = values.shape
        if g != self.n_genes:
            raise ValueError(f"expected {self.n_genes} genes, got {g}")
        summaries, token_parts = [], []
        for start in range(0, g, self.block_size):
            end = min(start + self.block_size, g)
            ids = torch.arange(start, end, device=values.device).unsqueeze(0).expand(b, -1)
            observed = (panel[:, start:end] > 0).long()
            tok = self.gene_emb(ids) + self.value_proj(values[:, start:end].unsqueeze(-1)) + self.mask_emb(observed)
            cls = self.block_cls.expand(b, -1, -1)
            h = self.local_encoder(torch.cat([cls, tok], dim=1))
            summaries.append(h[:, 0])
            token_parts.append(h[:, 1:])
        block_tokens = torch.stack(summaries, dim=1)
        global_cls = self.global_cls.expand(b, -1, -1)
        if self.use_resolution_token:
            if resolution_um is None:
                raise ValueError("resolution_um is required when use_resolution_token=True")
            r = torch.log2(resolution_um.float().view(b, 1, 1).clamp_min(1.0) / 8.0)
            global_cls = global_cls + self.resolution_proj(r)
        gh = self.global_encoder(torch.cat([global_cls, block_tokens], dim=1))
        z = torch.nn.functional.normalize(self.z_head(gh[:, 0]), dim=-1)
        pred = torch.cat([self.value_head(h).squeeze(-1) for h in token_parts], dim=1)
        return z, pred


class GeneTokenSparseFoundationTransformer(GeneTokenSparseTransformer):
    """Sparse encoder whose reconstruction target depends on the global state.

    The earlier sparse probe trained local token reconstruction only. Its
    global encoder and z_head received no reconstruction gradient. This
    variant keeps that checkpoint format separate and makes the global state
    part of every masked-gene prediction. A second-view representation loss
    in the trainer also directly trains z_head.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_to_local = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, self.d_model))

    def forward(self, values: torch.Tensor, panel: torch.Tensor, resolution_um: torch.Tensor | None = None):
        b, g = values.shape
        if g != self.n_genes:
            raise ValueError(f"expected {self.n_genes} genes, got {g}")
        summaries, token_parts = [], []
        for start in range(0, g, self.block_size):
            end = min(start + self.block_size, g)
            ids = torch.arange(start, end, device=values.device).unsqueeze(0).expand(b, -1)
            observed = (panel[:, start:end] > 0).long()
            tok = self.gene_emb(ids) + self.value_proj(values[:, start:end].unsqueeze(-1)) + self.mask_emb(observed)
            cls = self.block_cls.expand(b, -1, -1)
            h = self.local_encoder(torch.cat([cls, tok], dim=1))
            summaries.append(h[:, 0])
            token_parts.append(h[:, 1:])
        block_tokens = torch.stack(summaries, dim=1)
        global_cls = self.global_cls.expand(b, -1, -1)
        if self.use_resolution_token:
            if resolution_um is None:
                raise ValueError("resolution_um is required when use_resolution_token=True")
            r = torch.log2(resolution_um.float().view(b, 1, 1).clamp_min(1.0) / 8.0)
            global_cls = global_cls + self.resolution_proj(r)
        gh = self.global_encoder(torch.cat([global_cls, block_tokens], dim=1))
        z = torch.nn.functional.normalize(self.z_head(gh[:, 0]), dim=-1)
        global_context = self.global_to_local(gh[:, 0]).unsqueeze(1)
        pred = torch.cat([self.value_head(h + global_context).squeeze(-1) for h in token_parts], dim=1)
        return z, pred


def attention_backend_state() -> dict:
    out = {"sdpa_available": hasattr(torch.nn.functional, "scaled_dot_product_attention")}
    if torch.cuda.is_available():
        out.update({
            "flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
            "mem_efficient_sdp_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
            "math_sdp_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
            "device": torch.cuda.get_device_name(0),
        })
    else:
        out["device"] = "cpu"
    return out
