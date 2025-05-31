#!/bin/bash

export SAVE_DIR="$(pwd)/hacky"
mkdir -p ${SAVE_DIR}
export INPUT="${SAVE_DIR}/hacky_prob_net.csv"
export OUTPUT="${SAVE_DIR}/output_hacky_cell.csv"

export INTEGRATION_MANNER="bin"
export BUF_APPROACH="Hack-y"

export TIMEMLBuf="${SAVE_DIR}/hacky_time.csv"
export TIMERSZ="${SAVE_DIR}/hacky_rsz.csv"

# Save buffered trees
export PROJ_DIR=`pwd | grep -o "/\S*/MLBuf"`
export PROJ_DIR="${PROJ_DIR}/flows"

openroad run_replace_hacky.tcl | tee run_replace_hacky.log