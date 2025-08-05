set proj_dir "$::env(PROJ_DIR)"
set lib_setup_file lib_setup.tcl
set design_setup_file design_setup.tcl

set start [clock seconds]
source $lib_setup_file
source $design_setup_file

foreach lef_file ${lefs} {
  read_lef $lef_file
}

foreach lib_file ${libworst} {
  read_liberty $lib_file
}

read_def $floorplan_def
read_sdc $sdc
source $rc_file

remove_buffers
set_dont_use {CLKBUF_* TBUF_* BUF_X1}


global_placement 

set end [clock seconds]
puts "\[INFO\] Running time:   [expr $end - $start] second"

remove_buffers

write_def ${DESIGN}_no_timing.def
write_db ${DESIGN}_no_timing.odb

exit


