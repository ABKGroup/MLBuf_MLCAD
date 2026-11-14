set DESIGN swerv_wrapper
# ORFS nangate45/swerv_wrapper run up to IO placement (3_2_place_iop); clock period 2.0 ns (ORFS default constraint.sdc).
# The .odb carries the LEFs, so no read_lef is needed.
set sdc ${proj_dir}/OR_inputs/${DESIGN}/2_floorplan.sdc
set floorplan_odb ${proj_dir}/OR_inputs/${DESIGN}/3_2_place_iop.odb
# Global placement target density: the value ORFS uses for this design (PLACE_DENSITY_LB_ADDON 0.10 over the lower bound). The gpl default 0.70 fails global routing on swerv_wrapper.
set density 0.434234277009964
