#!/bin/bash

# Save buffered trees
export SAVE_DIR="$(pwd)/dataset"
export PROJ_DIR=`pwd | grep -o "/\S*/MLBuf"`
export PROJ_DIR="${PROJ_DIR}/flows"

openroad run_extract_or.tcl | tee extract_or.log
