#!/bin/bash
export PROJ_DIR=$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)   # repository root
export OPENROAD_EXE="${OPENROAD_EXE:-${PROJ_DIR}/OR_branch_integration/OpenROAD/build/src/openroad}"
export MLBUF_PYTHON="${MLBUF_PYTHON:-python}"
export MODEL_NAME="mlbuf-v2"
export CUDAID="0"
export SAVE_DIR="$(pwd)/${MODEL_NAME}"
export MODEL="${PROJ_DIR}/results/model_dict/${MODEL_NAME}.pt"
export PATHMLBUF="${PROJ_DIR}/OR_branch_integration"

mkdir -p ${SAVE_DIR}
rm -f "${SAVE_DIR:?}"/pred_boxes_*.csv "${SAVE_DIR:?}"/pred_trees_*.csv
export INPUT="${SAVE_DIR}/mlbuf_prob_net.csv"
export OUTPUT="${SAVE_DIR}/output_mlbuf_cell.csv"
: > ${OUTPUT}

export INTEGRATION_MANNER="bin"
export BUF_APPROACH="MLBuf"


${OPENROAD_EXE} run_replace_mlbuf.tcl | tee run_replace_mlbuf_${MODEL_NAME}.log
