"""Construct per-level training samples from net records.

A positive net with GT max depth D yields levels l = 0..D-1 (bottom-up, as in the paper):
  depth_i = D - l ;  input nodes = driver + {nodes at depth_i} + {sinks at depth < depth_i}
  labels  = nodes at depth_i grouped by their parent buffer (at depth_i - 1): that buffer's type and location;
            every other node -> None.   The top level (depth_i = 1) has no buffers: it is the "stop" sample.
Level-0 node features are the OpenSTA values before buffering (pre file). Upper-level features are produced by
the same feedback estimator that inference uses, applied to the GT structure, so train and inference see the
same feature distribution. A negative net yields one level-0 sample with no clusters.
"""
from __future__ import annotations

import numpy as np

from mlbuf2.physics import FeedbackEstimator, BufferLib, hpwl, WIRE_CAP_FF_PER_UM, HPWL_TO_WIRE, MAX_DEPTH
from mlbuf2.sources import SOURCES

NODE_FEAT_DIM = 15
CTX_FEAT_DIM = 9
DRIVER, SINK, BUFSINK = 0, 1, 2
# The two dump generations on disk follow different rsz sizing policies (Apr-2025 fork: X2..X32, X16-heavy;
# Nov-2025 build: X2/X4/X8 only). The policy is a property of the OpenROAD build, so it is an explicit input.
OR_VERSION_FLAG = {"apr2025": 0.0, "may2025": 0.0, "nov2025": 1.0}


def version_flag(rec) -> float:
    ver = rec.get("or_version") or SOURCES.get(rec.get("source", ""), {}).get("or_version", "apr2025")
    return OR_VERSION_FLAG.get(ver, 0.0)


def _node_feats(xy, cap, in_slew, is_type, drv_vals, hp):
    """xy: (N,2) rel um (driver row 0,0); cap: (N,) fF; in_slew: (N,) ps; is_type: (N,) int; drv_vals: dict."""
    N = len(xy)
    f = np.zeros((N, NODE_FEAT_DIM), np.float32)
    f[np.arange(N), is_type] = 1.0
    f[:, 3] = xy[:, 0] / 100.0
    f[:, 4] = xy[:, 1] / 100.0
    f[:, 5] = (np.abs(xy[:, 0]) + np.abs(xy[:, 1])) / 100.0
    f[:, 6] = xy[:, 0] / (hp + 1.0)
    f[:, 7] = xy[:, 1] / (hp + 1.0)
    f[:, 8] = np.log1p(np.maximum(cap, 0))
    f[:, 9] = np.log1p(np.maximum(in_slew, 0))
    d = is_type == DRIVER
    f[d, 8] = 0; f[d, 9] = 0
    f[d, 10] = np.log1p(max(drv_vals["out_slew_ps"], 0))
    f[d, 11] = np.log1p(max(drv_vals["out_cap_ff"], 0))
    f[d, 12] = np.log1p(max(drv_vals["max_cap_ff"], 0))
    f[d, 13] = np.log1p(max(drv_vals["res_ohm"], 0))
    ratio = drv_vals["out_cap_ff"] / max(drv_vals["max_cap_ff"], 1e-3)
    f[d, 14] = min(max(ratio, 0.0), 10.0) / 10.0
    return f


def _ctx_feats(n_nodes, hp, capsum, load, drv_vals, level, ver_flag):
    ratio = drv_vals["out_cap_ff"] / max(drv_vals["max_cap_ff"], 1e-3)
    return np.array([np.log1p(n_nodes), np.log1p(hp), np.log1p(max(capsum, 0)), np.log1p(max(load, 0)),
                     min(max(ratio, 0.0), 10.0) / 10.0, float(ratio > 1.0), level / float(MAX_DEPTH),
                     np.log1p(max(drv_vals["out_slew_ps"], 0)), float(ver_flag)], np.float32)


def _sample(meta, xy_nodes, cap_nodes, slew_nodes, type_nodes, drv_vals, cluster, ctype, cloc, gate, level, n_levels,
            ver_flag=0.0):
    """Assemble one level sample; the driver is row 0."""
    xy = np.concatenate([np.zeros((1, 2), np.float32), xy_nodes], 0)
    cap = np.concatenate([[0.0], cap_nodes]).astype(np.float32)
    slew = np.concatenate([[0.0], slew_nodes]).astype(np.float32)
    types = np.concatenate([[DRIVER], type_nodes]).astype(np.int64)
    hp = hpwl(xy)
    capsum = float(cap_nodes.sum())
    load = capsum + WIRE_CAP_FF_PER_UM * HPWL_TO_WIRE * hp
    return {
        "x": _node_feats(xy, cap, slew, types, drv_vals, hp),
        "ctx": _ctx_feats(len(xy), hp, capsum, load, drv_vals, level, ver_flag),
        "xy": xy, "cap": cap, "hpwl": np.float32(hp),
        "cluster": np.concatenate([[-1], cluster]).astype(np.int64),
        "ctype": np.asarray(ctype, np.int64), "cloc": np.asarray(cloc, np.float32).reshape(-1, 2),
        "gate": np.float32(gate), "level": level, "n_levels": n_levels, "meta": meta,
    }


def levels_from_record(rec, est: FeedbackEstimator, lib: BufferLib, max_levels: int = 12):
    meta = (rec["source"], rec["design"], rec["trigger"], rec["net"], rec["kind"])
    vf = version_flag(rec)
    if rec["kind"] == "neg":
        n = len(rec["sink_xy"])
        return [_sample(meta, rec["sink_xy"], rec["sink_cap_ff"], rec["sink_slew_ps"], np.full(n, SINK),
                        rec["drv"], np.full(n, -1), np.zeros(0, np.int64), np.zeros((0, 2), np.float32),
                        0.0, 0, 1, vf)]

    # node tables
    ids = np.concatenate([rec["sink_id"], rec["buf_id"]])
    xy = np.concatenate([rec["sink_xy"], rec["buf_xy"]], 0).astype(np.float32)
    depth = np.concatenate([rec["sink_depth"], rec["buf_depth"]])
    parent = np.concatenate([rec["sink_parent"], rec["buf_parent"]])
    is_buf = np.concatenate([np.zeros(len(rec["sink_id"]), bool), np.ones(len(rec["buf_id"]), bool)])
    cap = np.concatenate([rec["sink_cap_ff"], lib.in_cap_ff[rec["buf_type"]]]).astype(np.float32)
    btype = {int(b): int(t) for b, t in zip(rec["buf_id"], rec["buf_type"])}
    bxy = {int(b): xy[len(rec["sink_id"]) + i] for i, b in enumerate(rec["buf_id"])}
    D = int(depth.max())
    n_levels = min(D, max_levels)
    out = []
    for level in range(n_levels):
        depth_i = D - level
        sel = (depth == depth_i) | ((depth < depth_i) & ~is_buf)
        idx = np.where(sel)[0]
        if len(idx) == 0:
            break
        cur = depth[idx] == depth_i
        par = parent[idx]
        # cluster by parent buffer for current-depth nodes
        is_clustered = cur & np.array([p in btype for p in par])
        cl = np.full(len(idx), -1, np.int64)
        ctype, cloc = [], []
        if is_clustered.any():
            uniq = []
            for p in par[is_clustered]:
                if p not in uniq:
                    uniq.append(int(p))
            pos = {p: k for k, p in enumerate(uniq)}
            cl[is_clustered] = [pos[int(p)] for p in par[is_clustered]]
            ctype = [btype[p] for p in uniq]
            cloc = [bxy[p] for p in uniq]
        node_type = np.where(is_buf[idx], BUFSINK, SINK)
        if level == 0:
            drv_vals = dict(rec["drv"])
            slew = np.full(len(idx), drv_vals["out_slew_ps"], np.float32)
        else:
            drv_vals = dict(rec["drv"])
            oc = est.out_cap(cap[idx], xy[idx])
            os_ = est.out_slew(drv_vals["res_ohm"], oc, xy[idx])
            drv_vals["out_cap_ff"], drv_vals["out_slew_ps"], drv_vals["fanout"] = oc, os_, len(idx)
            slew = np.full(len(idx), os_, np.float32)
        out.append(_sample(meta, xy[idx], cap[idx], slew, node_type, drv_vals, cl, ctype, cloc, 1.0, level, n_levels, vf))
    return out
