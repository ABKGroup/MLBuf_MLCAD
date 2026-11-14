set libdir "${proj_dir}/OR_inputs/nangate45/lib"
set qrcdir "${proj_dir}/OR_inputs/nangate45"

# NanGate45 standard cells (the design has no macros).
set libworst "
  ${libdir}/NangateOpenCellLibrary_typical.lib \
  "

set rc_file "${qrcdir}/setRC.tcl"
