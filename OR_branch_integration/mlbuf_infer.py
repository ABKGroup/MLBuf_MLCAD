#!/usr/bin/env python3
"""MLBuf v2 inference hook of MLBuf-RePlAce.

The placer calls this script at every timing-driven trigger:
    $MLBUF_PYTHON mlbuf_infer.py --model <model.pt> --input <problem_nets.csv> --output <boxes.csv>
                                 --clusterNum <n> --pathMLBuf <this directory> --cudaID <n>

Input : the problem nets written by the placer (driver + sinks per net; um, s, F, ohm).
        The placer writes this file only when the trigger has problem nets, so a missing, empty or header-only
        file means "no problem nets": the hook then writes an empty box file and exits 0 (it never aborts the
        placement). After reading, the hook truncates the file when it is the placer's $INPUT file, so a later
        trigger without problem nets can never re-use this trigger's nets.
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
    MLBUF_KEEP_INPUT  1 keeps the problem-net file after reading (default: truncate it when it is $INPUT)
    SAVE_DIR          if set, the problem nets, boxes and buffer trees of every trigger are kept there as
                      prob_nets_<trigger>.csv, pred_boxes_<trigger>.csv and pred_trees_<trigger>.csv
"""
import argparse
import csv
import glob
import os
import shutil
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # repository root (mlbuf2 package)

TREE_HEADER = ["net", "level", "node_id", "node_type", "pin", "x_um", "y_um", "parent_id", "buffer_type"]


def env(name, default=None):
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def read_problem_nets(path):
    """Parsed problem nets of this trigger; [] when the file is missing, empty or header-only (not an error)."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return []
    import pandas as pd
    from mlbuf2 import schema
    try:
        return list(schema.parse_pre_file(path).values())
    except pd.errors.EmptyDataError:
        return []


def is_placer_input(path):
    """True when `path` is the placer's problem-net file ($INPUT). Only that file is truncated after reading,
    so running the hook by hand on a dump file never destroys the dump."""
    ref = env("INPUT")
    try:
        return bool(ref) and os.path.isfile(path) and os.path.samefile(path, ref)
    except OSError:
        return False


def consume_input(path):
    """Truncate the problem-net file: the placer rewrites it only when a trigger has problem nets, so without
    this a trigger with none would make the hook re-buffer the previous trigger's nets."""
    try:
        with open(path, "w"):
            pass
    except OSError as e:
        print(f"[MLBuf v2] warning: could not truncate {path}: {e}", file=sys.stderr)


def write_lines(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def write_tree(path, rows):
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(TREE_HEADER)
        wr.writerows(rows)


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
    t0 = time.time()

    # per-trigger archive (SAVE_DIR): the problem nets are copied before the input file is consumed
    save_dir, idx, trigger = env("SAVE_DIR"), None, ""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        idx = len(glob.glob(os.path.join(save_dir, "pred_boxes_*.csv")))
        trigger = f" trigger {idx}:"
        if os.path.isfile(a.input) and os.path.getsize(a.input) > 0:
            shutil.copyfile(a.input, os.path.join(save_dir, f"prob_nets_{idx:02d}.csv"))

    reason = "no input file" if not os.path.isfile(a.input) else "empty input file"
    recs = read_problem_nets(a.input)
    if is_placer_input(a.input) and env("MLBUF_KEEP_INPUT", "0") != "1":
        consume_input(a.input)

    if not recs:
        # nothing to buffer at this trigger: an empty box file, exit 0 (the model is not even loaded)
        write_lines(a.output, [])
        if save_dir:
            write_lines(os.path.join(save_dir, f"pred_boxes_{idx:02d}.csv"), [])
            write_tree(os.path.join(save_dir, f"pred_trees_{idx:02d}.csv"), [])
        print(f"[MLBuf v2]{trigger} no problem nets ({reason}); nets 0, buffered nets 0, buffers 0, "
              f"area 0.0 um^2, {time.time() - t0:.1f} s", flush=True)
        return 0

    import numpy as np
    import torch
    from mlbuf2.infer import Inferencer
    from mlbuf2.physics import DBU_PER_UM, MAX_DEPTH

    seed = int(env("MLBUF_SEED", "0"))
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(int(env("MLBUF_THREADS", "4")))
    gate_thr = float(env("MLBUF_GATE_THR")) if env("MLBUF_GATE_THR") else None
    depth = int(env("MLBUF_MAX_DEPTH", MAX_DEPTH))
    or_version = env("MLBUF_OR_VERSION", "apr2025")

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
    write_lines(a.output, lines)

    if save_dir:
        write_lines(os.path.join(save_dir, f"pred_boxes_{idx:02d}.csv"), lines)
        write_tree(os.path.join(save_dir, f"pred_trees_{idx:02d}.csv"), inf.tree_rows(states))

    print(f"[MLBuf v2]{trigger} nets {len(recs)}, buffered nets {sum(1 for s in states if s.buffers)}, "
          f"buffers {len(lines)}, area {area_um2:.1f} um^2, gate {inf.gate_thr:.2f}, device {inf.dev}, "
          f"{time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
