# This script was written and developed by ABKGroup students at UCSD. However, the underlying commands and reports are copyrighted by Cadence.
# We thank Cadence for granting permission to share our research to help promote and foster the next generation of innovators.
#!/bin/bash
module unload genus
module load genus/21.1
module unload innovus
module load innovus/21.1

#
# To run the Physical Synthesis (iSpatial) flow - flow
# Please enter the subdirectory of the directory that includes the design_setup.tcl and lib_setup.tcl files
export SEEDED_DEF=xx
export PHY_SYNTH=0
export PROJ_DIR=`pwd | grep -o "/\S*/MLBuf"`
#export PROJ_DIR="${PROJ_DIR}/flows"

mkdir log -p
innovus -64 -overwrite -log log/innovus.log -files ${PROJ_DIR}/scripts/invs_scripts/cadence/run_invs_evaluation.tcl
