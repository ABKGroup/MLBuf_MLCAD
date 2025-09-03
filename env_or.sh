#!/usr/bin/env bash
# ================================================================
#  OpenROAD / ORFS – Environment Setup Script
#
#  This script loads all required modules to reproduce the
#  environment used for running the project.
#
#  Usage:
#    source env_or.sh
# ================================================================

module unload git
module load git/2.9.5
module unload cmake
module load cmake/3.25.1
module unload gcc
module load gcc/9.3.0
module load flex/2.6.4
module load binutils/2.27
module load spdlog/1.8.1
module load swig/4.0.0
module unload tcl
module load tcl/8.6.6
module load lemon/1.3.1
module unload ortools
module load tclreadline/v2.1.0
module load ortools/9.4-c8
module unload klayout
module load klayout/0.27.1-c8
module load llvm/10.0.1
module load yosys
module load eqy
module unload boost
module load boost/1.81.0
module unload bison
module load bison/3.8.2
module load doxygen/1.8.15
module unload gtest
module load gtest/1.13.0
module load cudd
module load cuda/12.2
