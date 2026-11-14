#!/usr/bin/env python3
"""MLBuf v2 inference hook of MLBuf-RePlAce.

The placer calls this script at every timing-driven trigger:
    $MLBUF_PYTHON mlbuf_infer.py --model <model.pt> --input <problem_nets.csv> --output <boxes.csv>
                                 --clusterNum <n> --pathMLBuf <this directory> --cudaID <n>

Input : the problem nets written by the placer (driver + sinks per net; um, s, F, ohm).
Output: one line per predicted buffer, "lx,ly,ux,uy,area" in microns (um, um^2), no header.
        The placer converts the boxes to DBU and reserves their area in the density map.
        A net the model abstains on produces no line.
Inference is deterministic (argmax assignment, no sampling) and falls back to CPU when CUDA is unavailable.

Optional environment variables:
    MLBUF_GATE_THR    net gate threshold (default: the value stored in the checkpoint)
    MLBUF_MAX_DEPTH   recursion cap (default 5)
    MLBUF_OR_VERSION  rsz sizing policy of the OpenROAD build: apr2025 (default, OpenROAD_MLBuf) | nov2025
    MLBUF_SEED        random seed (default 0)
    MLBUF_THREADS     CPU threads of the inference (default 4); the predicted locations can differ in the
                      last digit between thread counts
    SAVE_DIR          if set, the boxes and buffer trees of every trigger are kept there as
                      pred_boxes_<trigger>.csv and pred_trees_<trigger>.csv
"""
import argparse
import csv
import glob
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # repository root (mlbuf2 package)


def env(name, default=None):
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def main():
    ap = argparse.ArgumentParser(description="MLBuf v2 inference hook.")
    ap.add_argument("--model", required=True, help="checkpoint (.pt)")
    ap.add_argument("--input", required=True, help="problem-net CSV written by the placer")
    ap.add_argument("--output", required=True, help="buffer-box CSV read by the placer (microns)")
    ap.add_argument("--clusterNum", default=None, help="accepted for compatibility; not used by MLBuf v2")
    ap.add_argument("--cudaID", default="0", help="CUDA device index")
    ap.add_argument("--pathMLBuf", default=HERE, help="accepted for compatibility")
    a = ap.parse_args()

    warnings.filterwarnings("ignore", message="enable_nested_tensor")
    import numpy as np
    import torch
    from mlbuf2 import schema
    from mlbuf2.infer import Inferencer
    from mlbuf2.physics import DBU_PER_UM, MAX_DEPTH

    t0 = time.time()
    seed = int(env("MLBUF_SEED", "0"))
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(int(env("MLBUF_THREADS", "4")))
    gate_thr = float(env("MLBUF_GATE_THR")) if env("MLBUF_GATE_THR") else None
    depth = int(env("MLBUF_MAX_DEPTH", MAX_DEPTH))
    or_version = env("MLBUF_OR_VERSION", "apr2025")

    recs = list(schema.parse_pre_file(a.input).values())
    kw = dict(gate_thr=gate_thr, max_depth=depth, or_version=or_version)
    try:
        inf = Inferencer(a.model, device=f"cuda:{int(a.cudaID)}", **kw)
    except RuntimeError as e:                       # CUDA reported available but unusable
        if "CUDA" not in str(e):
            raise
        inf = Inferencer(a.model, device="cpu", **kw)
    states = inf.run(recs)

    # The model places each buffer on the DBU grid; the placer expects microns.
    lines, area_um2 = [], 0.0
    for box in inf.boxes(states):
        lx, ly, ux, uy = (float(v) / DBU_PER_UM for v in box.split(",")[:4])
        area = (ux - lx) * (uy - ly)
        area_um2 += area
        lines.append(f"{lx:.4f},{ly:.4f},{ux:.4f},{uy:.4f},{area:.4f}")
    with open(a.output, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))

    trigger = ""
    save_dir = env("SAVE_DIR")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        idx = len(glob.glob(os.path.join(save_dir, "pred_boxes_*.csv")))
        trigger = f" trigger {idx}:"
        with open(os.path.join(save_dir, f"pred_boxes_{idx:02d}.csv"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        with open(os.path.join(save_dir, f"pred_trees_{idx:02d}.csv"), "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["net", "level", "node_id", "node_type", "pin", "x_um", "y_um", "parent_id", "buffer_type"])
            wr.writerows(inf.tree_rows(states))

    print(f"[MLBuf v2]{trigger} nets {len(recs)}, buffered nets {sum(1 for s in states if s.buffers)}, "
          f"buffers {len(lines)}, area {area_um2:.1f} um^2, gate {inf.gate_thr:.2f}, device {inf.dev}, "
          f"{time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
