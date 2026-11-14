set libdir "${proj_dir}/OR_inputs/nangate45/lib"
set qrcdir "${proj_dir}/OR_inputs/nangate45"

# NanGate45 standard cells only. The design's fakeram macro libraries (ADDITIONAL_LIBS of the ORFS config.mk,
# kept in the ORFS design directory) are not read here; the reported results were produced this way and the
# ORFS flow reads them downstream. Reading them here changes the timing-driven placement.
set libworst "
  ${libdir}/NangateOpenCellLibrary_typical.lib \
  "

set rc_file "${qrcdir}/setRC.tcl"
