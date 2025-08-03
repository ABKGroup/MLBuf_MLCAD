set libdir "${proj_dir}/inputs/ng45/lib"
set lefdir "${proj_dir}/inputs/ng45/lef"
set qrcdir "${proj_dir}/inputs/ng45/qrc"

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

set rc_file "${qrcdir}/setRC.tcl"
