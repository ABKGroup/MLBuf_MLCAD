"""Repository paths and training-data sources of MLBuf v2.

A source is a directory of rsz buffered-tree dumps (buffered_trees_{pre,mid,post}_<trigger>.csv), one file triple
per timing-driven trigger of a global placement run.

    dir           directory holding the dumps
    design        design name; the train/validation split is made by net within a design
    or_version    rsz sizing policy of the OpenROAD build that produced the dumps
                  ('apr2025' = OpenROAD_MLBuf fork: BUF_X2..X32; 'nov2025': BUF_X2/X4/X8 only)
    neg_triggers  triggers that contribute unbuffered nets as negatives ('all' or a list of trigger indices);
                  positives come from every trigger

To train on more designs, add their dump directories below (or point MLBUF_TRAINING_DATA at another root).
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # repository root
DATA = os.path.join(ROOT, "data")
TRAINING_DATA = os.environ.get("MLBUF_TRAINING_DATA") or os.path.join(DATA, "training_data")
PROCESSED = os.path.join(DATA, "processed")                                  # built by build_dataset.py
RESULTS = os.path.join(ROOT, "results")

SOURCES = {
    "ibex": dict(dir=os.path.join(TRAINING_DATA, "ibex"), design="ibex", or_version="apr2025", neg_triggers="all"),
    "jpeg": dict(dir=os.path.join(TRAINING_DATA, "jpeg"), design="jpeg", or_version="apr2025", neg_triggers="all"),
}

# NanGate45 buffer table (override with MLBUF_BUF_DATA)
BUF_DATA_CSV = os.environ.get("MLBUF_BUF_DATA") or os.path.join(DATA, "buf_data.csv")

# Negative subsampling for class balance: keep every unbuffered net with >= NEG_KEEP_MIN_SINKS sinks,
# and a deterministic NEG_KEEP_SMALL_FRAC share of the smaller ones.
NEG_KEEP_MIN_SINKS = 4
NEG_KEEP_SMALL_FRAC = 0.05
