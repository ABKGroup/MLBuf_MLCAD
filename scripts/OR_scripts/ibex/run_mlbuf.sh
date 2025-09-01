#!/bin/bash
export PROJ_DIR=$(pwd | grep -o "/\S*/MLBuf_MLCAD")
export MODEL_NAME="mlbuf-pretrained"
export CLUSTERNUM="20"
export CUDAID="0"
export SAVE_DIR="$(pwd)/${MODEL_NAME}"
export MODEL="${PROJ_DIR}/results/model_dict/${MODEL_NAME}.pt"
export PATHMLBUF="${PROJ_DIR}/OR_branch_integration"

mkdir -p ${SAVE_DIR}
export INPUT="${SAVE_DIR}/mlbuf_prob_net.csv"
export OUTPUT="${SAVE_DIR}/output_mlbuf_cell.csv"

export INTEGRATION_MANNER="bin"
export BUF_APPROACH="MLBuf"


${PROJ_DIR}/OR_branch_integration/OpenROAD/build/src/openroad run_replace_mlbuf.tcl | tee run_replace_mlbuf_${MODEL_NAME}.log

