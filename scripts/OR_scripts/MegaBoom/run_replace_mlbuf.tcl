set proj_dir "$::env(PROJ_DIR)"
set save_dir "$::env(SAVE_DIR)"
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

set new_overflow_list {75 70 65 60 55 50 45 40 35 30 25 20 15 10}
global_placement -timing_driven -keep_resize_below_overflow 0  -timing_driven_net_reweight_overflow $new_overflow_list

set end [clock seconds]
puts "\[INFO\] Running time:   [expr $end - $start] second"

remove_buffers
write_def ${save_dir}/${DESIGN}_mlbuf.def
write_db ${save_dir}/${DESIGN}_mlbuf.odb

exit


