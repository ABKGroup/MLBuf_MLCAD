set libdir "${proj_dir}/OR_inputs/nangate45/lib"
set qrcdir "${proj_dir}/OR_inputs/nangate45"

# NanGate45 standard cells plus the design's macro libraries (ADDITIONAL_LIBS of the ORFS config.mk).
set libworst "
  ${libdir}/NangateOpenCellLibrary_typical.lib \
  ${libdir}/fakeram45_2048x39.lib \
  ${libdir}/fakeram45_256x34.lib \
  ${libdir}/fakeram45_64x21.lib \
  "

set rc_file "${qrcdir}/setRC.tcl"
