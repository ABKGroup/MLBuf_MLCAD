set DESIGN tinyRocket
# ORFS nangate45/tinyRocket run up to IO placement (3_2_place_iop); clock period 1.4 ns (OR_inputs/tinyRocket/constraint.sdc; the ORFS default is 1.2 ns).
# The .odb carries the LEFs, so no read_lef is needed.
set sdc ${proj_dir}/OR_inputs/${DESIGN}/2_floorplan.sdc
set floorplan_odb ${proj_dir}/OR_inputs/${DESIGN}/3_2_place_iop.odb
# Global placement target density: the value ORFS uses for this design (nangate45 platform default 0.30).
set density 0.3
