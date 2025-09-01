set libdir "${proj_dir}/OR_inputs/nangate45/lib"
set lefdir "${proj_dir}/OR_inputs/nangate45/lef"
set qrcdir "${proj_dir}/OR_inputs/nangate45"

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
