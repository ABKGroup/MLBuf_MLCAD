#!/bin/bash

export PROJ_DIR=$(pwd | grep -o "/\S*/MLBuf_MLCAD")
export BUF_APPROACH="rsz"

${PROJ_DIR}/OR_branch_integration/OpenROAD/build/src/openroad run_replace_rsz_virtual.tcl | tee run_replace_rsz_virtual.log
