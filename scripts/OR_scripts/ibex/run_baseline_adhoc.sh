#!/bin/bash
export PROJ_DIR=$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)   # repository root
export OPENROAD_EXE="${OPENROAD_EXE:-${PROJ_DIR}/OR_branch_integration/OpenROAD/build/src/openroad}"
export MLBUF_PYTHON="${MLBUF_PYTHON:-python}"
export SAVE_DIR="$(pwd)/adhoc"
mkdir -p ${SAVE_DIR}
export INPUT="${SAVE_DIR}/adhoc_prob_net.csv"
export OUTPUT="${SAVE_DIR}/output_adhoc_cell.csv"
: > ${OUTPUT}
export PATHMLBUF="${PROJ_DIR}/OR_branch_integration"

export INTEGRATION_MANNER="bin"
export BUF_APPROACH="Adhoc"


${OPENROAD_EXE} run_replace_adhoc.tcl | tee run_replace_adhoc.log
