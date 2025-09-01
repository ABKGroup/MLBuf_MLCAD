#!/bin/bash


export PROJ_DIR=$(pwd | grep -o "/\S*/MLBuf_MLCAD")
${PROJ_DIR}/OR_branch_integration/OpenROAD/build/src/openroad run_replace_no_timing.tcl | tee run_replace_no_timing.log
