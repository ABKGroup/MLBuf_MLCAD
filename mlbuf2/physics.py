"""Physical constants, buffer library table, load-to-size rule, and the feedback estimator.

All quantities in um, ps, fF, ohm. NanGate45, 2000 DBU per um.
"""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd

DBU_PER_UM = 2000.0
# rsz wire capacitance: 8.88758e-11 F/m -> fF/um
WIRE_CAP_FF_PER_UM = 8.88758e-11 * 1e15 / 1e6          # 0.0888758 fF/um
HPWL_TO_WIRE = 1.2                                     # wire length ~ 1.2 * HPWL (paper, Eq. 11)
MAX_WIRELENGTH_UM = 1449128 / DBU_PER_UM               # fork's max_length in DBU -> 724.6 um
MAX_DEPTH = 5

# Load-to-size rule fitted to rsz (reproduces 86% of its choices): class by driven load in fF
LOAD_TABLE_FF = [(11.5, 1), (30.1, 2), (49.2, 3), (133.0, 4)]   # (upper bound, class); above -> 5


def load_to_size(load_ff):
    """Vectorised: load (fF) -> class 1..5."""
    load = np.asarray(load_ff, dtype=np.float64)
    cls = np.full(load.shape, 5, dtype=np.int64)
    for ub, c in reversed(LOAD_TABLE_FF):
        cls[load <= ub] = c
    return cls


class BufferLib:
    """Buffer table from buf_data.csv: index 0 = None, 1..5 = BUF_X2..X32."""

    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self.names = list(df["buf_type"])
        self.area = df["area"].to_numpy(np.float64)                       # um^2
        self.in_cap_ff = df["capacitance"].to_numpy(np.float64) * 1e15    # fF
        self.max_cap_ff = df["max_capacitance"].to_numpy(np.float64) * 1e15
        self.res = df["res"].to_numpy(np.float64)
        self.width = df["width"].to_numpy(np.float64)
        self.height = df["height"].to_numpy(np.float64)

    def __len__(self):
        return len(self.names)


def hpwl(xy: np.ndarray) -> float:
    if len(xy) == 0:
        return 0.0
    return float(xy[:, 0].max() - xy[:, 0].min() + xy[:, 1].max() - xy[:, 1].min())


class FeedbackEstimator:
    """Driver output cap and slew after a change of its direct children (paper Eq. 11-12, correct units,
    coefficients fitted on mid-file snapshots by calibrate_feedback.py).

        out_cap  = sum(child in_cap) + a_cap * WIRE_CAP * HPWL_TO_WIRE * HPWL(children + driver)
        out_slew = b_slew * (R_drv * out_cap) * 1e-3   [ohm*fF = 1e-3 ps]  + c_slew * HPWL * out_cap * 1e-3 + d_slew
    """

    def __init__(self, a_cap=1.0, b_slew=1.0, c_slew=0.0, d_slew=0.0):
        self.a_cap, self.b_slew, self.c_slew, self.d_slew = float(a_cap), float(b_slew), float(c_slew), float(d_slew)

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            return cls(**json.load(f)["coef"])

    def save(self, path: str, meta: dict | None = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"coef": dict(a_cap=self.a_cap, b_slew=self.b_slew, c_slew=self.c_slew, d_slew=self.d_slew),
                       "meta": meta or {}}, f, indent=2)

    def out_cap(self, child_cap_ff: np.ndarray, child_xy_rel: np.ndarray) -> float:
        pts = np.concatenate([child_xy_rel, np.zeros((1, 2), np.float32)], 0)  # driver at origin
        return float(child_cap_ff.sum()) + self.a_cap * WIRE_CAP_FF_PER_UM * HPWL_TO_WIRE * hpwl(pts)

    def out_slew(self, res_ohm: float, out_cap_ff: float, child_xy_rel: np.ndarray) -> float:
        pts = np.concatenate([child_xy_rel, np.zeros((1, 2), np.float32)], 0)
        h = hpwl(pts)
        return max(0.0, self.b_slew * res_ohm * out_cap_ff * 1e-3 + self.c_slew * h * out_cap_ff * 1e-3 + self.d_slew)

    def features(self, res_ohm, child_cap_ff, child_xy_rel):
        """Design-matrix rows used by the calibration (kept here so train/calibration share one definition)."""
        pts = np.concatenate([child_xy_rel, np.zeros((1, 2), np.float32)], 0)
        h = hpwl(pts)
        wire = WIRE_CAP_FF_PER_UM * HPWL_TO_WIRE * h
        return float(child_cap_ff.sum()), wire, h
