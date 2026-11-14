set DESIGN dynamic_node
# ORFS nangate45/dynamic_node run up to IO placement (3_2_place_iop); clock period 5.5 ns (OR_inputs/dynamic_node/constraint.sdc; the ORFS default is 6.0 ns).
# The .odb carries the LEFs, so no read_lef is needed.
set sdc ${proj_dir}/OR_inputs/${DESIGN}/2_floorplan.sdc
set floorplan_odb ${proj_dir}/OR_inputs/${DESIGN}/3_2_place_iop.odb
# Global placement target density: the value ORFS uses for this design (PLACE_DENSITY_LB_ADDON 0.20 over the lower bound).
set density 0.5431003904342652
