"""Losses: net gate, co-membership assignment, Hungarian-matched type / location, count, Chamfer (vectorised)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from mlbuf2.model import N_TYPES


def gate_loss(out, batch, pos_weight: float):
    sel = batch["level"] == 0
    if sel.sum() == 0:
        return out["gate"].sum() * 0
    return F.binary_cross_entropy_with_logits(out["gate"][sel], batch["gate"][sel],
                                              pos_weight=torch.tensor(pos_weight, device=out["gate"].device))


def comembership_loss(out, batch):
    """S = P P^T vs Y = same GT cluster, over node pairs where at least one node is clustered."""
    P, cl, mask = out.get("P_soft", out["P"]), batch["cluster"], batch["mask"]
    S = torch.einsum("bik,bjk->bij", P, P).clamp(1e-6, 1 - 1e-6)
    clustered = cl >= 0
    Y = (cl.unsqueeze(1) == cl.unsqueeze(2)) & clustered.unsqueeze(1) & clustered.unsqueeze(2)
    valid = mask.unsqueeze(1) & mask.unsqueeze(2) & (clustered.unsqueeze(1) | clustered.unsqueeze(2))
    eye = torch.eye(P.shape[1], dtype=torch.bool, device=P.device).unsqueeze(0)
    valid = valid & ~eye
    if valid.sum() == 0:
        return S.sum() * 0
    bce = -(Y.float() * torch.log(S) + (~Y).float() * torch.log(1 - S))
    pos, neg = valid & Y, valid & ~Y
    lp = bce[pos].mean() if pos.any() else S.sum() * 0
    ln = bce[neg].mean() if neg.any() else S.sum() * 0
    return 0.5 * (lp + ln)


def _scale(batch, cap=None):
    """um; location errors are measured relative to this (a quarter of the net's HPWL + 5 um, optionally capped
    so that large nets are still held to an absolute accuracy)."""
    s = batch["hpwl"] * 0.25 + 5.0
    return s.clamp(max=cap) if cap else s


@torch.no_grad()
def hungarian_match(out, batch, lam_loc: float = 1.0, scale_cap=None):
    """Match slots (K) to GT clusters (C) per sample. Returns index tensors (mb, mk, mc) on the batch device."""
    P, loc, cl, cmask, cloc = out["P"], out["loc"], batch["cluster"], batch["cmask"], batch["cloc"]
    B, N, K = P.shape
    C = cmask.shape[1]
    onehot = F.one_hot(cl.clamp(min=0), C).float() * (cl >= 0).unsqueeze(-1).float()      # [B,N,C]
    size = onehot.sum(1).clamp(min=1)
    overlap = torch.einsum("bnk,bnc->bkc", P, onehot) / size.unsqueeze(1)
    dist = (loc.unsqueeze(2) - cloc.unsqueeze(1)).abs().sum(-1) / _scale(batch, scale_cap).view(B, 1, 1)
    cost = (-overlap + lam_loc * dist).cpu().numpy()
    ncl = cmask.sum(1).tolist()
    mb, mk, mc = [], [], []
    for b in range(B):
        c = int(ncl[b])
        if c == 0:
            continue
        r, col = linear_sum_assignment(cost[b, :, :c])
        mb.extend([b] * len(r)); mk.extend(r.tolist()); mc.extend(col.tolist())
    dev = P.device
    return (torch.tensor(mb, dtype=torch.long, device=dev), torch.tensor(mk, dtype=torch.long, device=dev),
            torch.tensor(mc, dtype=torch.long, device=dev))


def type_loc_count_losses(out, batch, matches, class_weights, w_none: float = 0.3, scale_cap=None):
    """Matched slots: CE on type + smooth-L1 on location (HPWL-normalised). Unmatched slots: CE toward None.
    Count: expected #buffers vs #GT clusters. Chamfer: predicted buffers <-> GT buffers (both directions)."""
    tl, loc = out["type_logits"], out["loc"]
    ctype, cloc, cmask = batch["ctype"], batch["cloc"], batch["cmask"]
    B, K, _ = tl.shape
    dev = tl.device
    cw = class_weights.to(dev)
    mb, mk, mc = matches
    z = tl.sum() * 0
    scale = _scale(batch, scale_cap)
    if len(mb):
        t_loss = F.cross_entropy(tl[mb, mk], ctype[mb, mc], weight=cw, label_smoothing=0.05)
        l_loss = F.smooth_l1_loss(loc[mb, mk] / scale[mb, None], cloc[mb, mc] / scale[mb, None])
    else:
        t_loss, l_loss = z, z
    matched = torch.zeros(B, K, dtype=torch.bool, device=dev)
    if len(mb):
        matched[mb, mk] = True
    un = ~matched
    n_loss = F.cross_entropy(tl[un], torch.zeros(int(un.sum()), dtype=torch.long, device=dev), weight=cw,
                             label_smoothing=0.05) if un.any() else z
    p_buf = 1.0 - F.softmax(tl, -1)[..., 0]                                                   # [B,K]
    n_gt = cmask.sum(1).float()
    count_loss = (F.smooth_l1_loss(p_buf.sum(1), n_gt, reduction="none") / n_gt.clamp(min=1)).mean()
    has = n_gt > 0
    if has.any():
        d = (loc.unsqueeze(2) - cloc.unsqueeze(1)).abs().sum(-1) / scale.view(B, 1, 1)         # [B,K,C]
        d = d.masked_fill(~cmask.unsqueeze(1), 1e4)
        d_gt = (d + (1 - p_buf).unsqueeze(-1) * 10.0).min(1).values                              # [B,C]
        d_gt = (d_gt * cmask.float()).sum(1) / n_gt.clamp(min=1)
        d_pr = (p_buf * d.min(2).values).sum(1) / (p_buf.sum(1) + 1e-6)
        chamfer = (d_gt + d_pr)[has].mean()
    else:
        chamfer = z
    return {"type": t_loss, "loc": l_loss, "none": w_none * n_loss, "count": count_loss, "chamfer": chamfer}


def compute_class_weights(samples, clip=(0.2, 5.0)):
    cnt = np.ones(N_TYPES)
    for s in samples:
        for t in s["ctype"]:
            cnt[int(t)] += 1
    cnt[0] = cnt[1:].sum()
    w = cnt.sum() / (N_TYPES * cnt)
    w = np.clip(w / w[1:].mean(), *clip)
    w[0] = 1.0
    return torch.tensor(w, dtype=torch.float32)
