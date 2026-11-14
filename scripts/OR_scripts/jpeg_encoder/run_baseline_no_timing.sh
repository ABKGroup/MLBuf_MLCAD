#!/bin/bash

export PROJ_DIR=$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)   # repository root
export OPENROAD_EXE="${OPENROAD_EXE:-${PROJ_DIR}/OR_branch_integration/OpenROAD/build/src/openroad}"

${OPENROAD_EXE} run_replace_no_timing.tcl | tee run_replace_no_timing.log
