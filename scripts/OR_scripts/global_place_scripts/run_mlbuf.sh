#!/bin/bash

export MODEL_NAME="mlbuf_example"
export CLUSTERNUM="20"
export CUDAID="1"
export SAVE_DIR="$(pwd)/mlbuf_${MODEL_NAME}"
export MODEL="/home/dgx_projects/MLBuf/virtual_buffer/MLBuf/results/model_dict/MLBuf_${MODEL_NAME}data.pt"


mkdir -p ${SAVE_DIR}
export INPUT="${SAVE_DIR}/mlbuf_prob_net.csv"
export OUTPUT="${SAVE_DIR}/output_mlbuf_cell.csv"

export INTEGRATION_MANNER="bin"
export BUF_APPROACH="MLBuf"

export TIMEMLBuf="${SAVE_DIR}/mlbuf_time.csv"
export TIMERSZ="${SAVE_DIR}/mlbuf_rsz.csv"

# Save buffered trees
export PROJ_DIR=`pwd | grep -o "/\S*/MLBuf"`
export PROJ_DIR="${PROJ_DIR}/flows"


openroad run_replace_mlbuf.tcl | tee run_replace_mlbuf_${MODEL_NAME}.log

