#!/bin/bash

export PROJ_DIR=$(pwd | grep -o "/\S*/MLBuf_MLCAD")
export OPENROAD_EXE="${OPENROAD_EXE:-${PROJ_DIR}/OR_branch_integration/OpenROAD/build/src/openroad}"
export BUF_APPROACH="none"

${OPENROAD_EXE} run_replace_rsz_virtual.tcl | tee run_replace_rsz_virtual.log
