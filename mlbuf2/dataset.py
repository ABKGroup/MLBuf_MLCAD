"""Torch dataset over level samples, split manifests, normalisation, length-bucketed batching."""
from __future__ import annotations

import json
import os
import pickle
import random
import zlib
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from mlbuf2.levels import levels_from_record, NODE_FEAT_DIM, CTX_FEAT_DIM
from mlbuf2.physics import FeedbackEstimator, BufferLib
from mlbuf2.sources import SOURCES, PROCESSED, DATA, BUF_DATA_CSV


# ----------------------------------------------------------------------------- splits
def net_split(design: str, net: str, val_frac: float = 0.15) -> str:
    """Deterministic net-level split inside a design: all triggers/re-dumps of a net share the split."""
    h = zlib.crc32(f"{design}::{net}".encode()) % 10000
    return "val" if h < val_frac * 10000 else "train"


def load_records(sources, kinds=("pos", "neg")):
    recs = []
    for s in sources:
        p = os.path.join(PROCESSED, f"{s}.pkl")
        with open(p, "rb") as f:
            d = pickle.load(f)
        recs.extend(r for r in d["nets"] if r["kind"] in kinds)
    return recs


class LevelDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.lengths = np.array([len(s["x"]) for s in samples])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def build_samples(records, est: FeedbackEstimator, lib: BufferLib, split: str | None, val_frac=0.15,
                  neg_per_pos_cap: float | None = None, seed=0):
    """Expand records to level samples, optionally filtered to one split. neg_per_pos_cap limits the number of
    negative nets to cap * (#positive nets) by deterministic subsampling (class balance)."""
    pos = [r for r in records if r["kind"] == "pos"]
    neg = [r for r in records if r["kind"] == "neg"]
    if split is not None:
        pos = [r for r in pos if net_split(r["design"], r["net"], val_frac) == split]
        neg = [r for r in neg if net_split(r["design"], r["net"], val_frac) == split]
    if neg_per_pos_cap is not None and len(neg) > neg_per_pos_cap * len(pos):
        rng = random.Random(seed)
        rng.shuffle(neg)
        neg = neg[: int(neg_per_pos_cap * len(pos))]
    samples = []
    for r in pos + neg:
        samples.extend(levels_from_record(r, est, lib))
    return samples


# ----------------------------------------------------------------------------- normalisation
def compute_norm(samples):
    x = np.concatenate([s["x"] for s in samples], 0)
    c = np.stack([s["ctx"] for s in samples], 0)
    return {"x_mean": x.mean(0).tolist(), "x_std": (x.std(0) + 1e-6).tolist(),
            "c_mean": c.mean(0).tolist(), "c_std": (c.std(0) + 1e-6).tolist()}


# ----------------------------------------------------------------------------- batching
class BucketBatchSampler(Sampler):
    """Batches of similar node counts; bounded tokens per batch; shuffled batch order."""

    def __init__(self, lengths, max_tokens=12000, max_batch=96, shuffle=True, seed=0):
        self.lengths = np.asarray(lengths)
        self.max_tokens, self.max_batch, self.shuffle, self.seed = max_tokens, max_batch, shuffle, seed
        self.epoch = 0
        self.batches = self._make()

    def _make(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        order = np.argsort(self.lengths + (rng.rand(len(self.lengths)) * 2 if self.shuffle else 0), kind="stable")
        batches, cur, cur_max = [], [], 0
        for i in order:
            L = int(self.lengths[i])
            new_max = max(cur_max, L)
            if cur and (new_max * (len(cur) + 1) > self.max_tokens or len(cur) >= self.max_batch):
                batches.append(cur)
                cur, cur_max = [], 0
                new_max = L
            cur.append(int(i))
            cur_max = new_max
        if cur:
            batches.append(cur)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def set_epoch(self, e):
        self.epoch = e
        self.batches = self._make()

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def collate(samples, norm=None):
    B = len(samples)
    N = max(len(s["x"]) for s in samples)
    C = max([len(s["ctype"]) for s in samples] + [1])
    x = torch.zeros(B, N, NODE_FEAT_DIM)
    ctx = torch.zeros(B, CTX_FEAT_DIM)
    xy = torch.zeros(B, N, 2)
    cap = torch.zeros(B, N)
    mask = torch.zeros(B, N, dtype=torch.bool)          # valid node
    cluster = torch.full((B, N), -1, dtype=torch.long)
    ctype = torch.zeros(B, C, dtype=torch.long)
    cloc = torch.zeros(B, C, 2)
    cmask = torch.zeros(B, C, dtype=torch.bool)
    gate = torch.zeros(B)
    level = torch.zeros(B, dtype=torch.long)
    hp = torch.zeros(B)
    for b, s in enumerate(samples):
        n = len(s["x"])
        x[b, :n] = torch.from_numpy(s["x"])
        ctx[b] = torch.from_numpy(s["ctx"])
        xy[b, :n] = torch.from_numpy(s["xy"])
        cap[b, :n] = torch.from_numpy(s["cap"])
        mask[b, :n] = True
        cluster[b, :n] = torch.from_numpy(s["cluster"])
        c = len(s["ctype"])
        if c:
            ctype[b, :c] = torch.from_numpy(s["ctype"])
            cloc[b, :c] = torch.from_numpy(s["cloc"])
            cmask[b, :c] = True
        gate[b] = float(s["gate"])
        level[b] = int(s["level"])
        hp[b] = float(s["hpwl"])
    if norm is not None:
        x = (x - torch.tensor(norm["x_mean"])) / torch.tensor(norm["x_std"])
        x[~mask] = 0
        ctx = (ctx - torch.tensor(norm["c_mean"])) / torch.tensor(norm["c_std"])
    return {"x": x, "ctx": ctx, "xy": xy, "cap": cap, "mask": mask, "cluster": cluster, "ctype": ctype,
            "cloc": cloc, "cmask": cmask, "gate": gate, "level": level, "hpwl": hp,
            "meta": [s["meta"] for s in samples]}


def default_lib():
    return BufferLib(BUF_DATA_CSV)


def default_estimator():
    """Coefficients refitted by calibrate_feedback.py if present, else those of the pre-trained model."""
    for p in (os.path.join(PROCESSED, "feedback_estimator.json"), os.path.join(DATA, "feedback_estimator.json")):
        if os.path.exists(p):
            return FeedbackEstimator.load(p)
    return FeedbackEstimator()
