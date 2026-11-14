set proj_dir "$::env(PROJ_DIR)"
set lib_setup_file lib_setup.tcl
set design_setup_file design_setup.tcl

set start [clock seconds]
source $lib_setup_file
source $design_setup_file
set_thread_count 4

foreach lib_file ${libworst} {
  read_liberty $lib_file
}

read_db $floorplan_odb
read_sdc $sdc
source $rc_file

remove_buffers
set_dont_use {CLKBUF_* TBUF_* BUF_X1}

set new_overflow_list {75 70 65 60 55 50 45 40 35 30 25 20 15 10}
global_placement -density $density

set end [clock seconds]
puts "\[INFO\] Running time:   [expr $end - $start] second"

remove_buffers
write_def ${DESIGN}_no_timing.def
write_db ${DESIGN}_no_timing.odb

exit
