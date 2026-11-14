"""Repository paths and training-data sources of MLBuf v2.

A source is a directory of rsz buffered-tree dumps (buffered_trees_{pre,mid,post}_<trigger>.csv), one file triple
per timing-driven trigger of a global placement run.

    dir           directory holding the dumps
    design        design name; the train/validation split is made by net within a design
    or_version    rsz sizing policy of the OpenROAD build that produced the dumps
                  ('apr2025' = OpenROAD_MLBuf fork: BUF_X2..X32; 'nov2025': BUF_X2/X4/X8 only)
    neg_triggers  triggers that contribute unbuffered nets as negatives ('all' or a list of trigger indices);
                  positives come from every trigger

Source names are <design>_<OpenROAD build that produced the dumps>[_runN]; the build tag equals or_version.
The repository bundles ibex_apr2025 and jpeg_apr2025_run2 under data/training_data/. The complete data (all 11
training sources, see data/DATA.md) is downloaded separately; point MLBUF_TRAINING_DATA at the extracted directory.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # repository root
DATA = os.path.join(ROOT, "data")
TRAINING_DATA = os.environ.get("MLBUF_TRAINING_DATA") or os.path.join(DATA, "training_data")
PROCESSED = os.path.join(DATA, "processed")                                  # built by build_dataset.py
RESULTS = os.path.join(ROOT, "results")

SOURCES = {
    # Apr-2025 fork of OpenROAD (BUF_X2..X32 sizing)
    "ibex_apr2025":            dict(dir=os.path.join(TRAINING_DATA, "ibex_apr2025"),      design="ibex",          or_version="apr2025", neg_triggers="all"),
    "jpeg_apr2025_run2":       dict(dir=os.path.join(TRAINING_DATA, "jpeg_apr2025_run2"),      design="jpeg",          or_version="apr2025", neg_triggers="all"),
    "jpeg_apr2025_run1":       dict(dir=os.path.join(TRAINING_DATA, "jpeg_apr2025_run1"),      design="jpeg",          or_version="apr2025", neg_triggers=[0, 6, 12]),
    "ariane133_apr2025":       dict(dir=os.path.join(TRAINING_DATA, "ariane133_apr2025"), design="ariane133",     or_version="apr2025", neg_triggers=[0, 4, 8, 12]),
    "ariane133_may2025":       dict(dir=os.path.join(TRAINING_DATA, "ariane133_may2025"), design="ariane133",     or_version="may2025", neg_triggers=[2, 10]),
    "ariane136_apr2025":       dict(dir=os.path.join(TRAINING_DATA, "ariane136_apr2025"), design="ariane136",     or_version="apr2025", neg_triggers=[0, 4, 8, 12]),
    "black_parrot_apr2025":    dict(dir=os.path.join(TRAINING_DATA, "black_parrot_apr2025"),        design="bp",            or_version="apr2025", neg_triggers=[0, 6, 12]),
    "bsg_chip_apr2025":        dict(dir=os.path.join(TRAINING_DATA, "bsg_chip_apr2025"),  design="bsg_chip",      or_version="apr2025", neg_triggers=[0, 6, 12]),
    # Nov-2025 OpenROAD build (BUF_X2/X4/X8 sizing)
    "ibex_nov2025":            dict(dir=os.path.join(TRAINING_DATA, "ibex_nov2025"),       design="ibex",          or_version="nov2025", neg_triggers="all"),
    "jpeg_nov2025":            dict(dir=os.path.join(TRAINING_DATA, "jpeg_nov2025"),       design="jpeg",          or_version="nov2025", neg_triggers="all"),
    "ariane136_nov2025":       dict(dir=os.path.join(TRAINING_DATA, "ariane136_nov2025"),  design="ariane136",     or_version="nov2025", neg_triggers=[0, 6, 12]),
    # held-out design: never in train_sources; evaluation only
    "swerv_wrapper_nov2025":   dict(dir=os.path.join(TRAINING_DATA, "swerv_wrapper_nov2025"), design="swerv_wrapper", or_version="nov2025", neg_triggers="all"),
}

# NanGate45 buffer table (override with MLBUF_BUF_DATA)
BUF_DATA_CSV = os.environ.get("MLBUF_BUF_DATA") or os.path.join(DATA, "buf_data.csv")

# Negative subsampling for class balance: keep every unbuffered net with >= NEG_KEEP_MIN_SINKS sinks,
# and a deterministic NEG_KEEP_SMALL_FRAC share of the smaller ones.
NEG_KEEP_MIN_SINKS = 4
NEG_KEEP_SMALL_FRAC = 0.05
