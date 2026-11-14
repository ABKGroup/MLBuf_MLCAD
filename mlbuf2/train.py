"""Train MLBuf v2.   Usage: python -m mlbuf2.train configs/<name>.yaml [--gpu N] [--epochs E] [--quick]

Reads data/processed/<source>.pkl (build_dataset.py) and writes results/<name>/{model_best.pt, model_last.pt,
log.csv, best_val.json}. A checkpoint is self-contained: weights, model config, feature normalisation,
feedback-estimator coefficients and the validated gate threshold.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from mlbuf2.dataset import (LevelDataset, BucketBatchSampler, build_samples, collate, compute_norm,
                            default_estimator, default_lib, net_split)
from mlbuf2.losses import (compute_class_weights, comembership_loss, gate_loss, hungarian_match,
                           type_loc_count_losses)
from mlbuf2.model import MLBufV2
from mlbuf2.sources import PROCESSED, RESULTS


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_source(name, max_triggers=None):
    with open(os.path.join(PROCESSED, f"{name}.pkl"), "rb") as f:
        d = pickle.load(f)
    nets = d["nets"]
    if max_triggers is not None and max_triggers < len(d["triggers"]):
        keep = set(d["triggers"][:: max(1, len(d["triggers"]) // max_triggers)][:max_triggers])
        nets = [r for r in nets if r["trigger"] in keep]
    return nets


def to_dev(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def validate(model, loader, dev):
    """Level-wise metrics with hard assignment: gate F1 (level 0), count error, type accuracy on matched slots,
    location error (um), share of samples hitting the K ceiling."""
    model.eval()
    g_p, g_y = [], []
    cnt_err, cnt_rel, type_ok, type_n, loc_err, ceil = [], [], 0, 0, [], 0
    n_pos_levels = 0
    for batch in loader:
        batch = to_dev(batch, dev)
        out = model(batch)
        l0 = batch["level"] == 0
        g_p.append(torch.sigmoid(out["gate"][l0]).cpu()); g_y.append(batch["gate"][l0].cpu())
        tp = out["type_logits"].argmax(-1)
        pred_buf = (tp != 0) & (out["mass"] >= 1)                                   # [B,K]
        n_pred = pred_buf.sum(1).float()
        n_gt = batch["cmask"].sum(1).float()
        pos = batch["gate"] > 0
        cnt_err.append((n_pred - n_gt)[pos].abs().cpu()); cnt_rel.append(((n_pred - n_gt) / n_gt.clamp(min=1))[pos].cpu())
        ceil += int((n_pred >= model.K).sum()); n_pos_levels += int(pos.sum())
        mb, mk, mc = hungarian_match(out, batch)
        if len(mb):
            type_ok += int((tp[mb, mk] == batch["ctype"][mb, mc]).sum()); type_n += len(mb)
            loc_err.append((out["loc"][mb, mk] - batch["cloc"][mb, mc]).abs().sum(-1).cpu())
    g_p, g_y = torch.cat(g_p), torch.cat(g_y)
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.05, 0.95, 19):
        pr = g_p >= t
        tp_ = float((pr & (g_y > 0)).sum()); fp = float((pr & (g_y == 0)).sum()); fn = float((~pr & (g_y > 0)).sum())
        f1 = 2 * tp_ / max(2 * tp_ + fp + fn, 1)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    cnt_err = torch.cat(cnt_err) if cnt_err else torch.zeros(1)
    cnt_rel = torch.cat(cnt_rel) if cnt_rel else torch.zeros(1)
    loc_err = torch.cat(loc_err) if loc_err else torch.zeros(1)
    m = {"gate_f1": best_f1, "gate_thr": best_t, "count_mae": float(cnt_err.mean()),
         "count_bias": float(cnt_rel.mean()), "type_acc": type_ok / max(type_n, 1),
         "loc_med_um": float(loc_err.median()), "loc_mean_um": float(loc_err.mean()),
         "ceiling_frac": ceil / max(n_pos_levels, 1), "n_matched": type_n}
    m["score"] = m["gate_f1"] + m["type_acc"] + (1 - min(1.0, m["count_mae"] / 2.0)) + (1 - min(1.0, m["loc_med_um"] / 20.0))
    model.train()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--epochs", type=int)
    ap.add_argument("--quick", action="store_true", help="tiny subset for a smoke test")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.epochs:
        cfg["epochs"] = args.epochs
    set_seed(cfg.get("seed", 0))
    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(RESULTS, cfg["name"])
    os.makedirs(run_dir, exist_ok=True)
    lib, est = default_lib(), default_estimator()

    # ---------------- data
    t0 = time.time()
    recs = []
    for s in cfg["train_sources"]:
        recs.extend(load_source(s, cfg.get("max_triggers", {}).get(s)))
    if args.quick:
        random.Random(0).shuffle(recs); recs = recs[:3000]
    tr = build_samples(recs, est, lib, split="train", val_frac=cfg.get("val_frac", 0.15),
                       neg_per_pos_cap=cfg.get("neg_per_pos_cap", 1.5), seed=cfg.get("seed", 0))
    va = build_samples(recs, est, lib, split="val", val_frac=cfg.get("val_frac", 0.15),
                       neg_per_pos_cap=cfg.get("neg_per_pos_cap", 1.5), seed=cfg.get("seed", 0))
    del recs
    norm = compute_norm(tr)
    cw = compute_class_weights(tr, clip=tuple(cfg.get("class_weight_clip", [0.2, 5.0])))
    scap = cfg.get("loc_scale_cap")          # um; None = quarter-HPWL scale without a cap
    n_pos0 = sum(1 for s in tr if s["level"] == 0 and s["gate"] > 0); n_neg0 = sum(1 for s in tr if s["gate"] == 0)
    pos_weight = float(min(cfg.get("pos_weight_cap", 10.0), max(1.0, n_neg0 / max(n_pos0, 1))))
    info = {"train_samples": len(tr), "val_samples": len(va), "train_pos_nets": n_pos0, "train_neg_nets": n_neg0,
            "pos_weight": pos_weight, "class_weights": cw.tolist(), "build_sec": round(time.time() - t0),
            "train_nets_by_design": {}}
    for s in tr:
        if s["level"] == 0:
            info["train_nets_by_design"][s["meta"][1]] = info["train_nets_by_design"].get(s["meta"][1], 0) + 1
    print(json.dumps(info), flush=True)
    with open(os.path.join(run_dir, "data_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    with open(os.path.join(run_dir, "norm.json"), "w") as f:
        json.dump(norm, f)

    mt, mbs = cfg.get("max_tokens", 12000), cfg.get("max_batch", 96)
    tr_ds, va_ds = LevelDataset(tr), LevelDataset(va)
    tr_sampler = BucketBatchSampler(tr_ds.lengths, mt, mbs, shuffle=True, seed=cfg.get("seed", 0))
    va_sampler = BucketBatchSampler(va_ds.lengths, mt, mbs, shuffle=False)
    coll = lambda b: collate(b, norm)
    tr_loader = DataLoader(tr_ds, batch_sampler=tr_sampler, collate_fn=coll, num_workers=cfg.get("workers", 4),
                           persistent_workers=cfg.get("workers", 4) > 0)
    va_loader = DataLoader(va_ds, batch_sampler=va_sampler, collate_fn=coll, num_workers=2)

    # ---------------- model
    mcfg = cfg.get("model", {})
    model = MLBufV2(**mcfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 3e-4), weight_decay=cfg.get("weight_decay", 0.01))
    E = cfg["epochs"]; steps_per_epoch = len(tr_sampler); total = E * steps_per_epoch; warm = max(1, min(500, total // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    W = cfg.get("loss_weights", {})
    w = {k: W.get(k, 1.0) for k in ("gate", "assign", "type", "loc", "none", "count", "chamfer")}
    tau0, tau_min, tau_epochs = cfg.get("tau0", 1.0), cfg.get("tau_min", 0.3), cfg.get("tau_epochs", 0.4 * E)
    hard_after = cfg.get("hard_after", 0.6 * E)
    print(f"model params {sum(p.numel() for p in model.parameters())/1e6:.2f}M | steps/epoch {steps_per_epoch} | "
          f"pos_weight {pos_weight:.2f} | device {dev}", flush=True)

    log_path = os.path.join(run_dir, "log.csv"); best = -1e9; step = 0
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "tau", "hard", "lr", "loss", "gate", "assign", "type", "loc", "none", "count",
                                "chamfer", "val_score", "gate_f1", "gate_thr", "count_mae", "count_bias", "type_acc",
                                "loc_med_um", "ceiling_frac", "sec"])
    for epoch in range(E):
        tr_sampler.set_epoch(epoch)
        tau = max(tau_min, tau0 * (tau_min / tau0) ** min(1.0, epoch / max(tau_epochs, 1)))
        hard = epoch >= hard_after
        model.train(); t0 = time.time(); agg = {}; n = 0
        for batch in tr_loader:
            batch = to_dev(batch, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.get("amp", True)):
                out = model(batch, tau=tau, hard=hard)
            out = {k: (v.float() if torch.is_tensor(v) else v) for k, v in out.items()}
            L = {"gate": gate_loss(out, batch, pos_weight), "assign": comembership_loss(out, batch)}
            L.update(type_loc_count_losses(out, batch, hungarian_match(out, batch, scale_cap=scap), cw, w_none=1.0,
                                           scale_cap=scap))
            loss = sum(w[k] * v for k, v in L.items())
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1
            for k, v in L.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            agg["loss"] = agg.get("loss", 0.0) + float(loss); n += 1
        agg = {k: v / max(n, 1) for k, v in agg.items()}
        m = validate(model, va_loader, dev)
        sec = time.time() - t0
        row = [epoch, round(tau, 3), int(hard), sched.get_last_lr()[0], agg["loss"]] + [agg[k] for k in ("gate", "assign", "type", "loc", "none", "count", "chamfer")] + \
              [m["score"], m["gate_f1"], m["gate_thr"], m["count_mae"], m["count_bias"], m["type_acc"], m["loc_med_um"], m["ceiling_frac"], round(sec)]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([round(x, 4) if isinstance(x, float) else x for x in row])
        print(f"ep {epoch:3d} tau {tau:.2f} hard {int(hard)} loss {agg['loss']:.3f} "
              f"[g {agg['gate']:.3f} a {agg['assign']:.3f} t {agg['type']:.3f} l {agg['loc']:.3f} n {agg['none']:.3f} c {agg['count']:.3f} ch {agg['chamfer']:.3f}] "
              f"| val score {m['score']:.3f} gateF1 {m['gate_f1']:.3f}@{m['gate_thr']:.2f} cntMAE {m['count_mae']:.2f} bias {m['count_bias']:+.2f} "
              f"typeAcc {m['type_acc']:.3f} locMed {m['loc_med_um']:.1f}um ceil {m['ceiling_frac']:.3f} | {sec:.0f}s", flush=True)
        ckpt = {"model": model.state_dict(), "config": cfg, "norm": norm, "class_weights": cw.tolist(),
                "estimator": {"a_cap": est.a_cap, "b_slew": est.b_slew, "c_slew": est.c_slew, "d_slew": est.d_slew},
                "epoch": epoch, "val": m, "gate_thr": m["gate_thr"]}
        torch.save(ckpt, os.path.join(run_dir, "model_last.pt"))
        if m["score"] > best:
            best = m["score"]; torch.save(ckpt, os.path.join(run_dir, "model_best.pt"))
            with open(os.path.join(run_dir, "best_val.json"), "w") as f:
                json.dump({"epoch": epoch, **m}, f, indent=2)
    print("done. best val score", best, flush=True)


if __name__ == "__main__":
    main()
