"""Deterministic recursive inference for MLBuf v2.

Input : a problem-net CSV in the rsz dump schema (driver + sinks per net; the `pre` files or the `$INPUT`
        file written by the OpenROAD hook), or already-parsed records.
Output: per net the list of predicted buffers (level, type, location) with their children, plus
        - boxes CSV for the placer: lx,ly,ux,uy,area  in DBU (area in DBU^2), one line per buffer
        - tree CSV: net, level, node_id, node_type, pin, x, y (um, absolute), parent_id, buffer_type

Usage: python -m mlbuf2.infer --model results/model_dict/mlbuf-v2.pt --input <csv> --output <boxes.csv>
                              [--tree <tree.csv>] [--gpu 0] [--gate-thr T] [--core lx ly ux uy] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from mlbuf2 import schema
from mlbuf2.dataset import collate
from mlbuf2.levels import _sample, SINK, BUFSINK, version_flag
from mlbuf2.model import MLBufV2
from mlbuf2.physics import FeedbackEstimator, BufferLib, DBU_PER_UM, MAX_DEPTH
from mlbuf2.sources import BUF_DATA_CSV


class NetState:
    """Working state of one net during the recursion."""

    def __init__(self, rec):
        self.rec = rec
        n = len(rec["sink_xy"])
        self.xy = rec["sink_xy"].astype(np.float32)
        self.cap = rec["sink_cap_ff"].astype(np.float32)
        self.slew = rec["sink_slew_ps"].astype(np.float32)
        self.types = np.full(n, SINK, np.int64)
        self.ids = [("sink", i) for i in range(n)]            # identity of each current node
        self.drv = dict(rec["drv"])
        self.buffers = []                                      # dicts: level, type, xy(rel), children ids, id
        self.level = 0
        self.active = True
        self.gate_p = None
        self.ceiling_hit = False


class Inferencer:
    def __init__(self, ckpt_path: str, device="cuda:0", gate_thr=None, max_depth=MAX_DEPTH, max_tokens=60000,
                 or_version="apr2025"):
        """or_version: rsz sizing policy of the OpenROAD build that will consume the boxes ('apr2025' = the
        OpenROAD_MLBuf fork, BUF_X2..X32; 'nov2025' = BUF_X2/X4/X8 only); used when a record carries none."""
        self.or_version = or_version
        ck = torch.load(ckpt_path, map_location="cpu")
        self.cfg, self.norm = ck["config"], ck["norm"]
        self.model = MLBufV2(**self.cfg.get("model", {}))
        self.model.load_state_dict(ck["model"]); self.model.eval()
        self.dev = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.dev)
        self.est = FeedbackEstimator(**ck["estimator"])
        self.lib = BufferLib(BUF_DATA_CSV)
        self.gate_thr = float(gate_thr) if gate_thr is not None else float(ck.get("gate_thr", 0.5))
        self.max_depth, self.max_tokens = max_depth, max_tokens
        self.K = self.model.K

    def _level_sample(self, st: NetState):
        drv = dict(st.drv)
        if st.level > 0:
            oc = self.est.out_cap(st.cap, st.xy)
            os_ = self.est.out_slew(drv["res_ohm"], oc, st.xy)
            drv["out_cap_ff"], drv["out_slew_ps"], drv["fanout"] = oc, os_, len(st.xy)
            slew = np.full(len(st.xy), os_, np.float32)
        else:
            slew = st.slew
        st.rec.setdefault("or_version", self.or_version)
        return _sample(None, st.xy, st.cap, slew, st.types, drv, np.full(len(st.xy), -1),
                       np.zeros(0, np.int64), np.zeros((0, 2), np.float32), 0.0, st.level, 0, version_flag(st.rec))

    @torch.no_grad()
    def _forward(self, samples):
        batch = collate(samples, self.norm)
        batch = {k: (v.to(self.dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = self.model(batch)
        return {k: v.cpu() for k, v in out.items() if torch.is_tensor(v)}

    def run(self, recs, log_every=0):
        states = [NetState(r) for r in recs]
        t0 = time.time()
        for level in range(self.max_depth):
            active = [s for s in states if s.active]
            if not active:
                break
            # bucket by size, bounded tokens per batch
            active.sort(key=lambda s: len(s.xy))
            i = 0
            while i < len(active):
                j, mx = i, 0
                while j < len(active) and max(mx, len(active[j].xy) + 1) * (j - i + 1) <= self.max_tokens:
                    mx = max(mx, len(active[j].xy) + 1); j += 1
                j = max(j, i + 1)
                chunk = active[i:j]; i = j
                out = self._forward([self._level_sample(s) for s in chunk])
                self._apply(chunk, out, level)
            if log_every:
                print(f"  level {level}: {sum(1 for s in states if s.active)} nets still active "
                      f"({time.time() - t0:.1f}s)", flush=True)
        return states

    def _apply(self, chunk, out, level):
        gate = torch.sigmoid(out["gate"]).numpy()
        assign = out["P"].argmax(-1).numpy()                     # [B,N]
        mass = out["mass"].numpy()
        tp = out["type_logits"].argmax(-1).numpy()                # [B,K]
        loc = out["loc"].numpy()                                   # [B,K,2]
        for b, st in enumerate(chunk):
            n = len(st.xy)
            if level == 0:
                st.gate_p = float(gate[b])
                if st.gate_p < self.gate_thr:
                    st.active = False
                    continue
            slots = [k for k in range(self.K) if mass[b, k] >= 1 and tp[b, k] != 0]
            if not slots:
                st.active = False
                continue
            if len(slots) >= self.K:
                st.ceiling_hit = True
            a = assign[b, 1:n + 1]                                 # slot per node (node 0 is the driver)
            merged = np.zeros(n, bool)
            new_xy, new_cap, new_types, new_ids = [], [], [], []
            for k in slots:
                members = np.where(a == k)[0]
                if len(members) == 0:
                    continue
                merged[members] = True
                bid = ("buf", len(st.buffers))
                st.buffers.append({"id": bid, "level": level, "type": int(tp[b, k]),
                                   "xy": loc[b, k].astype(np.float32), "children": [st.ids[m] for m in members]})
                new_xy.append(loc[b, k]); new_cap.append(self.lib.in_cap_ff[int(tp[b, k])])
                new_types.append(BUFSINK); new_ids.append(bid)
            keep = ~merged
            n_next = int(keep.sum()) + len(new_xy)
            if len(new_xy) == 0 or n_next >= n:
                st.active = False
                continue
            st.xy = np.concatenate([st.xy[keep], np.array(new_xy, np.float32).reshape(-1, 2)], 0)
            st.cap = np.concatenate([st.cap[keep], np.array(new_cap, np.float32)])
            st.types = np.concatenate([st.types[keep], np.array(new_types, np.int64)])
            st.ids = [st.ids[i] for i in np.where(keep)[0]] + new_ids
            st.level = level + 1
            if st.level >= self.max_depth:
                st.active = False

    # ------------------------------------------------------------------ outputs
    def boxes(self, states, core=None):
        """Placer boxes in DBU: lx,ly,ux,uy,area(DBU^2). core: (lx,ly,ux,uy) in um to clip centres."""
        lines = []
        for st in states:
            dx, dy = st.rec["drv_xy"]
            for bf in st.buffers:
                t = bf["type"]
                w, h = self.lib.width[t], self.lib.height[t]
                cx, cy = float(dx + bf["xy"][0]), float(dy + bf["xy"][1])
                if core is not None:
                    cx = min(max(cx, core[0] + w / 2), core[2] - w / 2)
                    cy = min(max(cy, core[1] + h / 2), core[3] - h / 2)
                lx, ly = int(round((cx - w / 2) * DBU_PER_UM)), int(round((cy - h / 2) * DBU_PER_UM))
                ux, uy = int(round((cx + w / 2) * DBU_PER_UM)), int(round((cy + h / 2) * DBU_PER_UM))
                lines.append(f"{lx},{ly},{ux},{uy},{(ux - lx) * (uy - ly)}")
        return lines

    def tree_rows(self, states):
        rows = []
        for st in states:
            r = st.rec
            dx, dy = r["drv_xy"]
            rows.append([r["net"], -1, "drv", "Driver", r.get("drv_pin", ""), dx, dy, "", "None"])
            parent_of = {}
            for bf in st.buffers:
                for ch in bf["children"]:
                    parent_of[ch] = f"buf{bf['id'][1]}"
            for i, pin in enumerate(r["sink_pin"]):
                x, y = r["sink_xy"][i]
                rows.append([r["net"], 0, f"sink{i}", "Sink", pin, dx + x, dy + y, parent_of.get(("sink", i), "drv"), "-1"])
            for bf in st.buffers:
                rows.append([r["net"], bf["level"] + 1, f"buf{bf['id'][1]}", "Buffer", "", dx + bf["xy"][0], dy + bf["xy"][1],
                             parent_of.get(bf["id"], "drv"), schema.ID_TO_BUF_TYPE[bf["type"]]])
        return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--input", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--tree"); ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--gate-thr", type=float)
    ap.add_argument("--core", type=float, nargs=4); ap.add_argument("--seed", type=int, default=int(os.environ.get("MLBUF_SEED", 0)))
    ap.add_argument("--max-depth", type=int, default=MAX_DEPTH)
    ap.add_argument("--or-version", default="apr2025", choices=["apr2025", "nov2025"],
                    help="sizing policy of the OpenROAD build consuming the boxes")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    t0 = time.time()
    recs = list(schema.parse_pre_file(a.input).values())
    inf = Inferencer(a.model, device=f"cuda:{a.gpu}", gate_thr=a.gate_thr, max_depth=a.max_depth, or_version=a.or_version)
    states = inf.run(recs)
    lines = inf.boxes(states, core=a.core)
    with open(a.output, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    if a.tree:
        import csv
        with open(a.tree, "w", newline="") as f:
            wr = csv.writer(f); wr.writerow(["net", "level", "node_id", "node_type", "pin", "x_um", "y_um", "parent_id", "buffer_type"])
            wr.writerows(inf.tree_rows(states))
    nb = sum(len(s.buffers) for s in states)
    print(json.dumps({"nets_in": len(recs), "nets_buffered": sum(1 for s in states if s.buffers), "buffers": nb,
                      "area_um2": float(sum(inf.lib.area[b["type"]] for s in states for b in s.buffers)),
                      "gate_thr": inf.gate_thr, "sec": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
