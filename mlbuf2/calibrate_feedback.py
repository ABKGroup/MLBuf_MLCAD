"""Fit the feedback estimator (driver out_cap / out_slew after a change of its children) on mid-file snapshots.

Every mid snapshot k of a net gives the driver's OpenSTA out_cap/out_slew together with its direct children
(depth 1) at that moment. We fit
    out_cap  = sum(child in_cap) + a_cap * (WIRE_CAP * 1.2 * HPWL)
    out_slew = b_slew * R*C*1e-3 + c_slew * HPWL*C*1e-3 + d_slew
by least squares on the training sources and report the fit errors.
Also checks the assumption "child in_slew == driver out_slew".
Output: data/processed/feedback_estimator.json, which then replaces data/feedback_estimator.json (the
        coefficients of the pre-trained model) in training. Refit only when training on additional designs.
Usage:  python -m mlbuf2.calibrate_feedback [source ...]   (default: every source in sources.py)
"""
from __future__ import annotations

import glob
import json
import os
import sys
import numpy as np

from mlbuf2 import schema
from mlbuf2.physics import FeedbackEstimator
from mlbuf2.sources import SOURCES, PROCESSED

MAX_FILES = 6          # mid files per source (they are large; snapshots repeat the same nets)


def collect(sources):
    rows = []
    est = FeedbackEstimator()
    for s in sources:
        files = sorted(glob.glob(os.path.join(SOURCES[s]["dir"], "buffered_trees_mid_*.csv")))[:MAX_FILES]
        for f in files:
            for snap in schema.parse_mid_driver_snapshots(f):
                if snap["out_cap_ff"] <= 0 or snap["out_slew_ps"] <= 0 or snap["res_ohm"] <= 0:
                    continue
                capsum, wire, h = est.features(snap["res_ohm"], snap["ch_cap_ff"], snap["ch_xy"])
                rows.append((capsum, wire, h, snap["res_ohm"], snap["out_cap_ff"], snap["out_slew_ps"],
                             float(np.mean(np.abs(snap["ch_slew_ps"] - snap["out_slew_ps"]) /
                                           max(snap["out_slew_ps"], 1e-6)))))
        print(f"  {s}: {len(rows)} snapshots so far", flush=True)
    return np.array(rows, np.float64)


def fit(rows):
    capsum, wire, h, R, C, S, _ = rows.T
    # out_cap: C - capsum = a_cap * wire  -> a_cap by least squares through origin
    a_cap = float(np.dot(wire, C - capsum) / max(np.dot(wire, wire), 1e-12))
    # out_slew: S = b*R*C*1e-3 + c*h*C*1e-3 + d
    X = np.stack([R * C * 1e-3, h * C * 1e-3, np.ones_like(C)], 1)
    coef, *_ = np.linalg.lstsq(X, S, rcond=None)
    return FeedbackEstimator(a_cap=a_cap, b_slew=coef[0], c_slew=coef[1], d_slew=coef[2])


def report(est, rows, label):
    capsum, wire, h, R, C, S, slew_eq = rows.T
    cap_pred = capsum + est.a_cap * wire
    slew_pred = np.maximum(0, est.b_slew * R * C * 1e-3 + est.c_slew * h * C * 1e-3 + est.d_slew)
    rel = lambda p, t: float(np.median(np.abs(p - t) / np.maximum(np.abs(t), 1e-6)))
    out = {"n": int(len(rows)),
           "cap_median_rel_err": rel(cap_pred, C), "cap_median_rel_err_no_wire": rel(capsum, C),
           "slew_median_rel_err": rel(slew_pred, S),
           "slew_median_rel_err_rawRC": rel(R * C * 1e-3, S),
           "child_slew_eq_driver_slew_median_relerr": float(np.median(slew_eq))}
    print(f"[{label}] " + json.dumps(out), flush=True)
    return out


def main(sources=None):
    sources = list(sources or SOURCES)
    print("collecting snapshots ...")
    rows_fit = collect(sources)
    est = fit(rows_fit)
    print("coef:", est.a_cap, est.b_slew, est.c_slew, est.d_slew)
    m_fit = report(est, rows_fit, "fit")
    out = os.path.join(PROCESSED, "feedback_estimator.json")
    est.save(out, meta={"fit_sources": sources, "fit": m_fit})
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1:])
