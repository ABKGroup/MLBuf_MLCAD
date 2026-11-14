"""MLBuf v2: set encoder -> net gate + slot assignment -> per-slot type and location, with explicit cluster physics.

Kept from v1: one net (level) = a set of nodes; soft slot assignment (Gumbel-softmax in training); per-cluster
type then location conditioned on the type; recursion handled outside (levels / infer).
Changed: input projection + transformer encoder with a context token; mass-normalised pooling; cluster
physics (count, cap sum, soft bounding box, HPWL, load, load-table prior) as explicit features; a net-level gate;
location predicted inside the cluster's bounding box; argmax (no noise) at eval.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlbuf2.levels import NODE_FEAT_DIM, CTX_FEAT_DIM
from mlbuf2.physics import WIRE_CAP_FF_PER_UM, HPWL_TO_WIRE, LOAD_TABLE_FF

N_TYPES = 6
N_PHYS = 9


def mlp(i, h, o, p=0.0):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, o))


def load_prior(load_ff: torch.Tensor) -> torch.Tensor:
    """load (fF) -> one-hot over 6 classes (index 0 unused) from the load table."""
    cls = torch.full_like(load_ff, 5, dtype=torch.long)
    for ub, c in reversed(LOAD_TABLE_FF):
        cls = torch.where(load_ff <= ub, torch.full_like(cls, c), cls)
    return F.one_hot(cls, N_TYPES).float()


class MLBufV2(nn.Module):
    def __init__(self, d=128, K=32, n_layers=3, n_heads=4, dropout=0.1, bbox_expand=0.1, soft_beta=0.2):
        super().__init__()
        self.d, self.K, self.bbox_expand, self.soft_beta = d, K, bbox_expand, soft_beta
        self.node_in = mlp(NODE_FEAT_DIM, d, d)
        self.ctx_in = mlp(CTX_FEAT_DIM, d, d)
        layer = nn.TransformerEncoderLayer(d, n_heads, 2 * d, dropout=dropout, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.enc_norm = nn.LayerNorm(d)
        self.gate_head = mlp(d, d, 1)
        self.assign_head = nn.Linear(d, K)
        self.slot_in = nn.Linear(d + N_PHYS + N_TYPES, d)
        slot_layer = nn.TransformerEncoderLayer(d, n_heads, 2 * d, dropout=dropout, batch_first=True,
                                                norm_first=True, activation="gelu")
        self.slot_attn = nn.TransformerEncoder(slot_layer, 1)
        self.slot_norm = nn.LayerNorm(d)
        self.type_head = mlp(d + N_PHYS + N_TYPES, d, N_TYPES)
        self.loc_head = mlp(d + N_PHYS + N_TYPES, d, 2)

    # ------------------------------------------------------------------ cluster physics (differentiable)
    def cluster_physics(self, P, xy, cap, mask):
        """P: [B,N,K] (rows of padding / driver are 0); xy: [B,N,2] um; cap: [B,N] fF."""
        eps = 1e-6
        m = P.sum(1)                                            # [B,K] mass
        capk = torch.einsum("bnk,bn->bk", P, cap)
        # soft bounding box: softmax over nodes of (beta * coord + log P)
        logP = torch.log(P + 1e-9)
        def soft_ext(coord, sign):
            w = torch.softmax(sign * self.soft_beta * coord.unsqueeze(-1) + logP, dim=1)   # [B,N,K]
            return torch.einsum("bnk,bn->bk", w, coord)
        xmax, xmin = soft_ext(xy[..., 0], 1.0), soft_ext(xy[..., 0], -1.0)
        ymax, ymin = soft_ext(xy[..., 1], 1.0), soft_ext(xy[..., 1], -1.0)
        empty = m < eps
        xmax, xmin, ymax, ymin = [torch.where(empty, torch.zeros_like(v), v) for v in (xmax, xmin, ymax, ymin)]
        cx = torch.einsum("bnk,bn->bk", P, xy[..., 0]) / (m + eps)
        cy = torch.einsum("bnk,bn->bk", P, xy[..., 1]) / (m + eps)
        hp = (xmax - xmin).clamp(min=0) + (ymax - ymin).clamp(min=0)
        load = capk + WIRE_CAP_FF_PER_UM * HPWL_TO_WIRE * hp
        phys = torch.stack([torch.log1p(m), torch.log1p(capk.clamp(min=0)), torch.log1p(hp), torch.log1p(load.clamp(min=0)),
                            cx / 100, cy / 100, (xmax - xmin) / 100, (ymax - ymin) / 100,
                            (m > 0.5).float()], -1)
        bbox = torch.stack([xmin, ymin, xmax, ymax], -1)
        return m, capk, load, hp, phys, bbox

    def forward(self, batch, tau: float = 1.0, hard: bool = False):
        x, ctx, xy, cap, mask = batch["x"], batch["ctx"], batch["xy"], batch["cap"], batch["mask"]
        B, N, _ = x.shape
        h = self.node_in(x)
        c = self.ctx_in(ctx).unsqueeze(1)
        seq = torch.cat([c, h], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), ~mask], 1)
        z = self.enc_norm(self.encoder(seq, src_key_padding_mask=pad))
        zc, zn = z[:, 0], z[:, 1:]
        gate_logit = self.gate_head(zc).squeeze(-1)

        logits = self.assign_head(zn)                                        # [B,N,K]
        amask = mask.clone()
        amask[:, 0] = False                                                  # the driver is never merged
        if self.training:
            g = -torch.empty_like(logits).exponential_().log()               # Gumbel(0,1) noise
            P_soft = F.softmax((logits + g) / tau, dim=-1)
            if hard:
                P_hard = F.one_hot(P_soft.argmax(-1), self.K).float()
                P = P_hard - P_soft.detach() + P_soft                         # straight-through
            else:
                P = P_soft
        else:
            P = F.one_hot(logits.argmax(-1), self.K).float()
            P_soft = F.softmax(logits, dim=-1)
        am = amask.unsqueeze(-1).float()
        P, P_soft = P * am, P_soft * am

        m, capk, load, hp, phys, bbox = self.cluster_physics(P, xy, cap, mask)
        emb = torch.einsum("bnk,bnd->bkd", P, zn) / (m.unsqueeze(-1) + 1e-6)
        prior = load_prior(load.detach())
        s = self.slot_in(torch.cat([emb, phys, prior], -1))
        s = self.slot_norm(self.slot_attn(s, src_key_padding_mask=(m < 1e-6)))
        s = torch.nan_to_num(s)
        type_logits = self.type_head(torch.cat([s, phys, prior], -1))       # [B,K,6]
        uv = torch.sigmoid(self.loc_head(torch.cat([s, phys, F.softmax(type_logits, -1)], -1)))
        ext = (bbox[..., 2:] - bbox[..., :2]).clamp(min=1.0)                 # [B,K,2] (>= 1 um)
        lo = bbox[..., :2] - self.bbox_expand * ext
        loc = lo + uv * (1 + 2 * self.bbox_expand) * ext                     # [B,K,2] um rel. to driver
        return {"gate": gate_logit, "P": P, "P_soft": P_soft, "logits": logits, "mass": m,
                "type_logits": type_logits, "loc": loc, "load": load, "hpwl_k": hp, "bbox": bbox}
