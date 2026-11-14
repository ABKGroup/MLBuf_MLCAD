"""Parsing of rsz buffered-tree CSV dumps (all schema versions seen on disk) into per-net records.

Dump files per trigger:
  buffered_trees_pre_<ts>.csv   every net rsz visited: driver + sinks, OpenSTA values before any buffering
  buffered_trees_post_<ts>.csv  final tree of every net that received >= 1 buffer
  buffered_trees_mid_<ts>.csv   one snapshot per inserted buffer (Tree ID = buffer order; tree 0 = unbuffered)

Column sets: 21 (no reason), 22 (+ Buffer Reason), 24 (+ wire_length, total_wirelength), 23/24 (+ Driver Slack,
Wire Delay), optionally + File Name. Names may carry leading spaces. Units in the dumps: um, s, F, ohm.

Record units: um (x, y, absolute and relative to the driver), ps, fF, ohm.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

BUF_TYPE_TO_ID = {"None": 0, "-1": 0, "BUF_X2": 1, "BUF_X4": 2, "BUF_X8": 3, "BUF_X16": 4, "BUF_X32": 5}
ID_TO_BUF_TYPE = {0: "None", 1: "BUF_X2", 2: "BUF_X4", 3: "BUF_X8", 4: "BUF_X16", 5: "BUF_X32"}

NUMERIC = ["Tree ID", "Depth", "Node ID", "X", "Y", "Input Slew", "Output Slew", "Input Cap", "Output Cap",
           "Max Output Cap", "Fanout", "Resistance", "Sink Count", "Buffer Count", "#Children", "Buffer Order"]

S_TO_PS = 1e12
F_TO_FF = 1e15


def read_tree_csv(path: str, tree0_only: bool = False) -> pd.DataFrame:
    """Read one dump; normalise column names and dtypes. Parent Node ID stays a string ('None' for the driver)."""
    df = pd.read_csv(path, low_memory=False, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    for c in NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Buffer Type" in df.columns:
        df["Buffer Type"] = df["Buffer Type"].str.strip()
    if tree0_only and "Tree ID" in df.columns:
        df = df[df["Tree ID"] == 0]
    return df


def trigger_id(path: str) -> str:
    """'buffered_trees_post_2025331_215116.csv' -> '2025331_215116'; '..._post_10.csv' -> '10'."""
    base = os.path.basename(path)
    for k in ("_pre_", "_mid_", "_post_"):
        if k in base:
            return base.split(k)[1].rsplit(".csv", 1)[0]
    return base


def _driver_values(drv: pd.Series) -> dict:
    return {
        "out_slew_ps": float(drv["Output Slew"]) * S_TO_PS,
        "out_cap_ff": float(drv["Output Cap"]) * F_TO_FF,
        "max_cap_ff": float(drv["Max Output Cap"]) * F_TO_FF,
        "res_ohm": float(drv["Resistance"]),
        "fanout": int(drv["Fanout"]) if not np.isnan(drv["Fanout"]) else 0,
    }


def parse_pre_net(g: pd.DataFrame):
    """One pre-file net group -> record with driver + sinks (no buffers). Returns None if malformed."""
    drv_rows = g[g["Node Type"] == "Driver"]
    snk = g[g["Node Type"] == "Sink"]
    if len(drv_rows) == 0 or len(snk) == 0:
        return None
    drv = drv_rows.iloc[0]
    dx, dy = float(drv["X"]), float(drv["Y"])
    rec = {
        "net": str(drv["Net Name"]),
        "drv_xy": np.array([dx, dy], np.float32),
        "drv": _driver_values(drv),
        "sink_pin": snk["Pin Name"].to_numpy(dtype=object),
        "sink_xy": np.stack([snk["X"].to_numpy(np.float32) - dx, snk["Y"].to_numpy(np.float32) - dy], 1),
        "sink_cap_ff": (snk["Input Cap"].to_numpy(np.float64) * F_TO_FF).astype(np.float32),
        "sink_slew_ps": (snk["Input Slew"].to_numpy(np.float64) * S_TO_PS).astype(np.float32),
    }
    return rec


def parse_post_net(g: pd.DataFrame):
    """One post-file net group -> record with the full tree (sinks with depth/parent, buffers with type/loc)."""
    drv_rows = g[g["Node Type"] == "Driver"]
    snk = g[g["Node Type"] == "Sink"]
    buf = g[g["Node Type"] == "Buffer"]
    if len(drv_rows) == 0 or len(snk) == 0:
        return None
    drv = drv_rows.iloc[0]
    dx, dy = float(drv["X"]), float(drv["Y"])

    def parent_ids(df):
        p = df["Parent Node ID"].to_numpy(dtype=object)
        return np.array([-1 if (str(v) == "None" or str(v) == "") else int(float(v)) for v in p], np.int64)

    rec = {
        "net": str(drv["Net Name"]),
        "drv_xy": np.array([dx, dy], np.float32),
        "drv_id": int(drv["Node ID"]),
        "post_drv": _driver_values(drv),           # values after buffering (reference only)
        "sink_id": snk["Node ID"].to_numpy(np.int64),
        "sink_pin": snk["Pin Name"].to_numpy(dtype=object),
        "sink_xy": np.stack([snk["X"].to_numpy(np.float32) - dx, snk["Y"].to_numpy(np.float32) - dy], 1),
        "sink_cap_ff": (snk["Input Cap"].to_numpy(np.float64) * F_TO_FF).astype(np.float32),
        "sink_depth": snk["Depth"].to_numpy(np.int64),
        "sink_parent": parent_ids(snk),
        "buf_id": buf["Node ID"].to_numpy(np.int64),
        "buf_xy": np.stack([buf["X"].to_numpy(np.float32) - dx, buf["Y"].to_numpy(np.float32) - dy], 1)
        if len(buf) else np.zeros((0, 2), np.float32),
        "buf_type": np.array([BUF_TYPE_TO_ID.get(t, 0) for t in buf["Buffer Type"]], np.int64),
        "buf_depth": buf["Depth"].to_numpy(np.int64),
        "buf_parent": parent_ids(buf),
        "buf_order": buf["Buffer Order"].to_numpy(np.int64) if "Buffer Order" in buf.columns else np.zeros(len(buf), np.int64),
    }
    if "Buffer Reason" in buf.columns:
        rec["buf_reason"] = buf["Buffer Reason"].str.strip().to_numpy(dtype=object)
    return rec


def iter_nets(df: pd.DataFrame):
    for net, g in df.groupby("Net Name", sort=False):
        yield net, g


def parse_post_file(path: str) -> dict:
    df = read_tree_csv(path)
    out = {}
    for net, g in iter_nets(df):
        rec = parse_post_net(g)
        if rec is not None and len(rec["buf_id"]) > 0:
            out[net] = rec
    return out


def parse_pre_file(path: str, keep=None) -> dict:
    """keep: optional callable(net_name, n_sinks) -> bool to subsample (used for negatives)."""
    df = read_tree_csv(path)
    out = {}
    if keep is None:
        for net, g in iter_nets(df):
            rec = parse_pre_net(g)
            if rec is not None:
                out[net] = rec
        return out
    sizes = df[df["Node Type"] == "Sink"].groupby("Net Name").size()
    wanted = {n for n, s in sizes.items() if keep(n, int(s))}
    df = df[df["Net Name"].isin(wanted)]
    for net, g in iter_nets(df):
        rec = parse_pre_net(g)
        if rec is not None:
            out[net] = rec
    return out


def parse_mid_driver_snapshots(path: str):
    """For feedback-estimator calibration: for every (net, tree k) snapshot return the driver values and its
    direct children (depth 1): x, y (rel), in_cap. Yields dicts."""
    df = read_tree_csv(path)
    for (net, tid), g in df.groupby(["Net Name", "Tree ID"], sort=False):
        drv_rows = g[g["Depth"] == 0]
        ch = g[g["Depth"] == 1]
        if len(drv_rows) == 0 or len(ch) == 0:
            continue
        drv = drv_rows.iloc[0]
        dx, dy = float(drv["X"]), float(drv["Y"])
        yield {
            "net": net, "tree": int(tid),
            "res_ohm": float(drv["Resistance"]),
            "out_cap_ff": float(drv["Output Cap"]) * F_TO_FF,
            "out_slew_ps": float(drv["Output Slew"]) * S_TO_PS,
            "ch_xy": np.stack([ch["X"].to_numpy(np.float32) - dx, ch["Y"].to_numpy(np.float32) - dy], 1),
            "ch_cap_ff": (ch["Input Cap"].to_numpy(np.float64) * F_TO_FF).astype(np.float32),
            "ch_is_buf": (ch["Node Type"] == "Buffer").to_numpy(),
            "ch_slew_ps": (ch["Input Slew"].to_numpy(np.float64) * S_TO_PS).astype(np.float32),
        }
