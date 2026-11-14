#!/bin/bash

export PROJ_DIR=$(pwd | grep -o "/\S*/MLBuf_MLCAD")
export OPENROAD_EXE="${OPENROAD_EXE:-${PROJ_DIR}/OR_branch_integration/OpenROAD/build/src/openroad}"

${OPENROAD_EXE} run_replace_no_timing.tcl | tee run_replace_no_timing.log
