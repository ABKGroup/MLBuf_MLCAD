#!/bin/bash
# Usage: scripts/OR_scripts/run_orfs.sh <DESIGN_NAME> <METHOD>
#   DESIGN_NAME: ibex | jpeg_encoder | swerv_wrapper | tinyRocket | dynamic_node
#   METHOD:      mlbuf | adhoc | no_timing | rsz_virtual
# Runs the ORFS flow (resizing, detailed placement, CTS, routing, final report) from the
# global placement result of METHOD and prints the post-route metrics.
# Set FLOW_HOME to OpenROAD-flow-scripts/flow before running.

DESIGN_NAME=$1
METHOD=$2
export PROJ_DIR=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
export OPENROAD_EXE="${OPENROAD_EXE:-${PROJ_DIR}/OR_branch_integration/OpenROAD/build/src/openroad}"
RUN_DIR="${PROJ_DIR}/scripts/OR_scripts/${DESIGN_NAME}"

# ORFS design directory and per-design make variables. Global placement is not rerun here, so the IO-pin and
# SDC variables below only keep the flow configuration identical to the one used for the reported results:
# swerv_wrapper's io.tcl uses exclude_io_pin_region, which this OpenROAD lacks (fork place_pins has -exclude);
# tinyRocket and dynamic_node use the clock periods in OR_inputs/<design>/constraint.sdc (1.4 ns and 5.5 ns)
# instead of the ORFS defaults (1.2 ns and 6.0 ns).
DESIGN_ARGS=()
case ${DESIGN_NAME} in
  ibex)          ORFS_DESIGN="ibex" ;;
  jpeg_encoder)  ORFS_DESIGN="jpeg" ;;
  swerv_wrapper) ORFS_DESIGN="swerv_wrapper"
                 DESIGN_ARGS=(IO_CONSTRAINTS= "PLACE_PINS_ARGS=-exclude left:* -exclude right:* -exclude top:* -exclude bottom:0-200 -exclude bottom:1000-1100") ;;
  tinyRocket)    ORFS_DESIGN="tinyRocket"
                 DESIGN_ARGS=("SDC_FILE=${PROJ_DIR}/OR_inputs/${DESIGN_NAME}/constraint.sdc") ;;
  dynamic_node)  ORFS_DESIGN="dynamic_node"
                 DESIGN_ARGS=("SDC_FILE=${PROJ_DIR}/OR_inputs/${DESIGN_NAME}/constraint.sdc") ;;
  *) echo "Unknown design: ${DESIGN_NAME}"; exit 1 ;;
esac
case ${METHOD} in
  mlbuf)       GP_ODB="${RUN_DIR}/mlbuf-v2/${DESIGN_NAME}_mlbuf.odb" ;;
  adhoc)       GP_ODB="${RUN_DIR}/${DESIGN_NAME}_rsz_adhoc.odb" ;;
  no_timing)   GP_ODB="${RUN_DIR}/${DESIGN_NAME}_no_timing.odb" ;;
  rsz_virtual) GP_ODB="${RUN_DIR}/${DESIGN_NAME}_rsz_virtual.odb" ;;
  *) echo "Unknown method: ${METHOD}"; exit 1 ;;
esac
[ -f "${GP_ODB}" ] || { echo "Global placement result not found: ${GP_ODB}"; exit 1; }
[ -f "${FLOW_HOME}/Makefile" ] || { echo "Set FLOW_HOME to OpenROAD-flow-scripts/flow"; exit 1; }

WORK_HOME="${RUN_DIR}/orfs_${METHOD}"
mkdir -p ${WORK_HOME}
# Global placement is not rerun; BUF_APPROACH is only read by timing-driven global placement.
export BUF_APPROACH="none"
cd ${FLOW_HOME}
make -f ${FLOW_HOME}/Makefile \
  DESIGN_CONFIG=${FLOW_HOME}/designs/nangate45/${ORFS_DESIGN}/config.mk \
  OPENROAD_EXE=${OPENROAD_EXE} FLOW_HOME=${FLOW_HOME} WORK_HOME=${WORK_HOME} FLOW_VARIANT=run \
  INPUTS_DIR=${PROJ_DIR}/OR_inputs/${DESIGN_NAME} GP_ODB=${GP_ODB} SKIP_GP_MLBUF=1 "${DESIGN_ARGS[@]}" \
  NUM_CORES=${NUM_CORES:-4} mlbuf_place_and_route > ${WORK_HOME}/flow.log 2>&1

LOG_DIR=$(ls -d ${WORK_HOME}/logs/nangate45/*/run 2>/dev/null | head -1)
if [ ! -f "${LOG_DIR}/6_report.json" ]; then
  echo "ORFS did not finish; see ${WORK_HOME}/flow.log"
  exit 1
fi
python3 - "${LOG_DIR}" <<'PY'
import json, sys
rep = json.load(open(sys.argv[1] + "/6_report.json"))
drt = json.load(open(sys.argv[1] + "/5_2_route.json"))
wl = [v for k, v in drt.items() if k.startswith("detailedroute__route__wirelength")]
print("WNS (ns): %.3f  TNS (ns): %.2f  power (mW): %.1f  rWL (um): %.0f" % (
    rep["finish__timing__setup__ws"], rep["finish__timing__setup__tns"],
    1e3 * rep["finish__power__total"], wl[-1]))
PY
