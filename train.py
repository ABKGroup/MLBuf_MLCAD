#!/usr/bin/env python3
"""Entry point for training MLBuf v2.

    python train.py [config] [--gpu N] [--epochs E] [--quick]

    config    training configuration (default: configs/mlbuf_v2.yaml)
    --quick   small subset of the data, for a smoke test

Steps:
  1. build the net records from the dumps in data/training_data (mlbuf2/build_dataset.py);
     skipped when data/processed/<source>.pkl already exists
  2. train the model (mlbuf2/train.py)
Results are written to results/<config name>/.
"""
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    args = sys.argv[1:]
    if not args or args[0].startswith("-"):
        args.insert(0, os.path.join(HERE, "configs", "mlbuf_v2.yaml"))
    with open(args[0]) as f:
        sources = yaml.safe_load(f)["train_sources"]

    from mlbuf2 import build_dataset, train
    from mlbuf2.sources import PROCESSED

    for s in sources:
        if not os.path.exists(os.path.join(PROCESSED, f"{s}.pkl")):
            build_dataset.build_source(s)

    sys.argv = ["train.py"] + args
    train.main()


if __name__ == "__main__":
    main()
