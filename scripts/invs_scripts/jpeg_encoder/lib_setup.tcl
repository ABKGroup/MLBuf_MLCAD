# This script was written and developed by ABKGroup students at UCSD. However, the underlying commands and reports are copyrighted by Cadence. 
# We thank Cadence for granting permission to share our research to help promote and foster the next generation of innovators.
# lib and lef, RC setup

set libdir "${proj_dir}/inputs/ng45/lib"
set lefdir "${proj_dir}/inputs/ng45/lef"
set qrcdir "${proj_dir}/inputs/ng45/qrc"

set_db init_lib_search_path { \
  ${libdir} \
  ${lefdir} \
}

set libworst "  
  ${libdir}/NangateOpenCellLibrary_typical.lib \
  "


set libbest " 
  ${libdir}/NangateOpenCellLibrary_typical.lib \
  "

set lefs "  
  ${lefdir}/NangateOpenCellLibrary.tech.lef \
  ${lefdir}/NangateOpenCellLibrary.macro.mod.lef \
  "

set qrc_max "${qrcdir}/NG45.tch"
set qrc_min "${qrcdir}/NG45.tch"
#
# Ensures proper and consistent library handling between Genus and Innovus
#set_db library_setup_ispatial true
setDesignMode -process 45
