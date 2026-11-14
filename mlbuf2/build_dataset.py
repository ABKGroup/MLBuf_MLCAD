"""Build per-source net records from the existing dumps.

For every trigger file pair (pre, post) of a source:
  positives : every net in post (tree structure, buffer types/locations) + the driver's pre-buffering OpenSTA
              values (out_slew, out_cap) taken from the pre file of the same trigger
  negatives : nets in pre and not in post (driver + sinks), subsampled by sink count for class balance

Output: data/processed/<source>.pkl  -> {"source", "design", "triggers": [...], "nets": [record, ...]}
Usage:  python -m mlbuf2.build_dataset [source ...]   (default: every source in sources.py)
"""
from __future__ import annotations

import glob
import os
import pickle
import sys
import time
import zlib
from multiprocessing import Pool

import numpy as np

from mlbuf2 import schema
from mlbuf2.sources import SOURCES, PROCESSED, NEG_KEEP_MIN_SINKS, NEG_KEEP_SMALL_FRAC


def _keep_negative(net: str, n_sinks: int) -> bool:
    if n_sinks >= NEG_KEEP_MIN_SINKS:
        return True
    return (zlib.crc32(net.encode()) % 10000) < NEG_KEEP_SMALL_FRAC * 10000


def _file_pairs(d: str):
    posts = sorted(glob.glob(os.path.join(d, "buffered_trees_post_*.csv")))
    pairs = []
    for p in posts:
        ts = schema.trigger_id(p)
        pre = os.path.join(d, f"buffered_trees_pre_{ts}.csv")
        pairs.append((ts, pre if os.path.exists(pre) else None, p))
    return pairs


def build_trigger(args):
    src, ts, pre_path, post_path, want_neg = args
    t0 = time.time()
    meta = SOURCES[src]
    pos = schema.parse_post_file(post_path)
    stats = {"trigger": ts, "post_nets": len(pos), "pre_nets": 0, "neg_kept": 0, "pos_without_pre": 0}
    nets = []
    if pre_path is not None:
        df = schema.read_tree_csv(pre_path)
        drv = df[df["Node Type"] == "Driver"].drop_duplicates("Net Name").set_index("Net Name")
        stats["pre_nets"] = len(drv)
        for net, rec in pos.items():
            if net in drv.index:
                rec["drv"] = schema._driver_values(drv.loc[net])
            else:
                rec["drv"] = dict(rec["post_drv"])
                stats["pos_without_pre"] += 1
        if want_neg:
            sizes = df[df["Node Type"] == "Sink"].groupby("Net Name").size()
            keep = {n for n, s in sizes.items() if n not in pos and _keep_negative(n, int(s))}
            dneg = df[df["Net Name"].isin(keep)]
            for net, g in schema.iter_nets(dneg):
                rec = schema.parse_pre_net(g)
                if rec is None:
                    continue
                rec.update(kind="neg", source=src, design=meta["design"], trigger=ts)
                nets.append(rec)
            stats["neg_kept"] = len(nets)
    else:
        for rec in pos.values():
            rec["drv"] = dict(rec["post_drv"])
            stats["pos_without_pre"] += 1
    for net, rec in pos.items():
        rec.update(kind="pos", source=src, design=meta["design"], trigger=ts)
        # unbuffered view: every sink sees the driver's pre-buffering output slew
        rec["sink_slew_ps"] = np.full(len(rec["sink_id"]), rec["drv"]["out_slew_ps"], np.float32)
        nets.append(rec)
    stats["sec"] = round(time.time() - t0, 1)
    return nets, stats


def build_source(src: str, workers: int = 13):
    meta = SOURCES[src]
    pairs = _file_pairs(meta["dir"])
    neg_idx = set(range(len(pairs))) if meta["neg_triggers"] == "all" else set(meta["neg_triggers"])
    jobs = [(src, ts, pre, post, i in neg_idx) for i, (ts, pre, post) in enumerate(pairs)]
    t0 = time.time()
    with Pool(min(workers, len(jobs))) as pool:
        results = pool.map(build_trigger, jobs)
    nets, stats = [], []
    for n, s in results:
        nets.extend(n)
        stats.append(s)
    os.makedirs(PROCESSED, exist_ok=True)
    out = os.path.join(PROCESSED, f"{src}.pkl")
    with open(out, "wb") as f:
        pickle.dump({"source": src, "design": meta["design"], "or_version": meta["or_version"],
                     "triggers": [p[0] for p in pairs], "stats": stats, "nets": nets}, f, protocol=5)
    npos = sum(1 for r in nets if r["kind"] == "pos")
    nneg = len(nets) - npos
    print(f"[{src}] triggers={len(pairs)} pos={npos} neg={nneg} "
          f"pos_without_pre={sum(s['pos_without_pre'] for s in stats)} "
          f"-> {out} ({time.time() - t0:.0f}s)", flush=True)
    return out


if __name__ == "__main__":
    names = sys.argv[1:] or list(SOURCES)
    for s in names:
        build_source(s)
