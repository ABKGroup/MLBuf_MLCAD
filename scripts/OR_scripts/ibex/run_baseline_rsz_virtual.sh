#!/bin/bash

# Save buffered trees
export PROJ_DIR=`pwd | grep -o "/\S*/MLBuf"`
#export PROJ_DIR="${PROJ_DIR}/flows"
openroad run_replace_rsz_virtual.tcl | tee run_replace_rsz_virtual.log
