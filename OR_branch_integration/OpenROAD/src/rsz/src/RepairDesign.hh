/////////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2025, The Regents of the University of California
// All rights reserved.
//
// BSD 3-Clause License
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <string>
#include "BufferedNet.hh"
#include "PreChecks.hh"
#include "db_sta/dbSta.hh"
#include "rsz/Resizer.hh"
#include "sta/Corner.hh"
#include "sta/Delay.hh"
#include "sta/GraphClass.hh"
#include "sta/Liberty.hh"
#include "sta/LibertyClass.hh"
#include "sta/NetworkClass.hh"
#include "utl/Logger.h"
#include "BufferedTree.h"

namespace rsz {

class Resizer;
enum class ParasiticsSrc;

using std::vector;

using sta::Corner;
using sta::dbNetwork;
using sta::dbSta;
using sta::LibertyCell;
using sta::LibertyPort;
using sta::MinMax;
using sta::Net;
using sta::Pin;
using sta::PinSeq;
using sta::Slew;
using sta::StaState;
using sta::Vertex;

using odb::Rect;

// Region for partioning fanout pins.
class LoadRegion
{
 public:
  LoadRegion();
  LoadRegion(PinSeq& pins, Rect& bbox);

  PinSeq pins_;
  Rect bbox_;  // dbu
  vector<LoadRegion> regions_;
};

class RepairDesign : dbStaState
{
 public:
  explicit RepairDesign(Resizer* resizer);
  ~RepairDesign() override;
  void repairDesign(double max_wire_length,
                    double slew_margin,
                    double cap_margin,
                    double buffer_gain,
                    bool verbose);
  // Update since 2025/01/27.

  // Defualt for store_buffered_trees_flag is false.
  void repairDesign(double max_wire_length,  // zero for none (meters)
                    double slew_margin,
                    double cap_margin,
                    double buffer_gain,
                    bool verbose,
                    int& repaired_net_count,
                    int& slew_violations,
                    int& cap_violations,
                    int& fanout_violations,
                    int& length_violations,
                    bool store_buffered_trees_flag = false);
  int insertedBufferCount() const { return inserted_buffer_count_; }
  void repairNet(Net* net,
                 double max_wire_length,
                 double slew_margin,
                 double cap_margin);
  void repairClkNets(double max_wire_length);
  void repairClkInverters();

  // Update since 2025/01/27
  void initBufferedTrees();
  void writeBufferedTrees(const std::string& file_name);

  // Update since 2025/01/27
  void repairDesign_update(
    double max_wire_length,  // zero for none (meters)
    double slew_margin, // zero for default
    double cap_margin, // zero for default
    double buffer_gain, // zero for default
    bool verbose,
    int& repaired_net_count,
    int& slew_violations,
    int& cap_violations,
    int& fanout_violations,
    int& length_violations);

 protected:
  void init();

  bool getCin(const Pin* drvr_pin, float& cin);
  void findBufferSizes();
  bool performGainBuffering(Net* net, const Pin* drvr_pin, int max_fanout);

  void repairNet(Net* net,
                 const Pin* drvr_pin,
                 Vertex* drvr,
                 bool check_slew,
                 bool check_cap,
                 bool check_fanout,
                 int max_length,  // dbu
                 bool resize_drvr,
                 int& repaired_net_count,
                 int& slew_violations,
                 int& cap_violations,
                 int& fanout_violations,
                 int& length_violations);
  bool needRepairSlew(const Pin* drvr_pin,
                      int& slew_violations,
                      float& max_cap,
                      const Corner*& corner);
  bool needRepairCap(const Pin* drvr_pin,
                     int& cap_violations,
                     float& max_cap,
                     const Corner*& corner);
  bool needRepairWire(int max_length, int wire_length, int& length_violations);
  bool needRepair(const Pin* drvr_pin,
                  const Corner*& corner,
                  int max_length,
                  int wire_length,
                  bool check_cap,
                  bool check_slew,
                  float& max_cap,
                  int& slew_violations,
                  int& cap_violations,
                  int& length_violations);
  bool checkLimits(const Pin* drvr_pin,
                   bool check_slew,
                   bool check_cap,
                   bool check_fanout);
  void checkSlew(const Pin* drvr_pin,
                 // Return values.
                 Slew& slew,
                 float& limit,
                 float& slack,
                 const Corner*& corner);
  float bufferInputMaxSlew(LibertyCell* buffer, const Corner* corner) const;
  void repairNet(const BufferedNetPtr& bnet,
                 const Pin* drvr_pin,
                 float max_cap,
                 int max_length,  // dbu
                 const Corner* corner);
  void repairNet(const BufferedNetPtr& bnet,
                 int level,
                 // Return values.
                 int& wire_length,
                 PinSeq& load_pins);
  void checkSlewLimit(float ref_cap, float max_load_slew);
  void repairNetWire(const BufferedNetPtr& bnet,
                     int level,
                     // Return values.
                     int& wire_length,
                     PinSeq& load_pins);
  void repairNetJunc(const BufferedNetPtr& bnet,
                     int level,
                     // Return values.
                     int& wire_length,
                     PinSeq& load_pins);
  void repairNetLoad(const BufferedNetPtr& bnet,
                     int level,
                     // Return values.
                     int& wire_length,
                     PinSeq& load_pins);
  float maxSlewMargined(float max_slew);
  double findSlewLoadCap(LibertyPort* drvr_port,
                         double slew,
                         const Corner* corner);
  double gateSlewDiff(LibertyPort* drvr_port,
                      double load_cap,
                      double slew,
                      const DcalcAnalysisPt* dcalc_ap);
  LoadRegion findLoadRegions(const Pin* drvr_pin, int max_fanout);
  void subdivideRegion(LoadRegion& region, int max_fanout);
  void makeRegionRepeaters(LoadRegion& region,
                           int max_fanout,
                           int level,
                           const Pin* drvr_pin,
                           bool check_slew,
                           bool check_cap,
                           int max_length,
                           bool resize_drvr);
  void makeFanoutRepeater(PinSeq& repeater_loads,
                          PinSeq& repeater_inputs,
                          const Rect& bbox,
                          const Point& loc,
                          bool check_slew,
                          bool check_cap,
                          int max_length,
                          bool resize_drvr);
  PinSeq findLoads(const Pin* drvr_pin);
  Rect findBbox(PinSeq& pins);
  Point findClosedPinLoc(const Pin* drvr_pin, PinSeq& pins);
  bool isRepeater(const Pin* load_pin);
  void makeRepeater(const char* reason,
                    const Point& loc,
                    LibertyCell* buffer_cell,
                    bool resize,
                    int level,
                    // Return values.
                    PinSeq& load_pins,
                    float& repeater_cap,
                    float& repeater_fanout,
                    float& repeater_max_slew);
  void makeRepeater(const char* reason,
                    int x,
                    int y,
                    LibertyCell* buffer_cell,
                    bool resize,
                    int level,
                    // Return values.
                    PinSeq& load_pins,
                    float& repeater_cap,
                    float& repeater_fanout,
                    float& repeater_max_slew,
                    Net*& out_net,
                    Pin*& repeater_in_pin,
                    Pin*& repeater_out_pin);
  
  LibertyCell* findBufferUnderSlew(float max_slew, float load_cap);
  bool hasInputPort(const Net* net);
  double dbuToMeters(int dist) const;
  int metersToDbu(double dist) const;

  void printProgress(int iteration,
                     bool force,
                     bool end,
                     int repaired_net_count) const;

  Logger* logger_ = nullptr;
  dbNetwork* db_network_ = nullptr;
  PreChecks* pre_checks_ = nullptr;
  Resizer* resizer_;
  int dbu_ = 0;
  ParasiticsSrc parasitics_src_ = ParasiticsSrc::none;

  // Gain buffering
  std::vector<LibertyCell*> buffer_sizes_;

  // Implicit arguments to repairNet bnet recursion.
  const Pin* drvr_pin_ = nullptr;
  float max_cap_ = 0;
  int max_length_ = 0;
  double slew_margin_ = 0;
  double cap_margin_ = 0;
  double buffer_gain_ = 0;
  const Corner* corner_ = nullptr;

  int resize_count_ = 0;
  int inserted_buffer_count_ = 0;
  const MinMax* min_ = MinMax::min();
  const MinMax* max_ = MinMax::max();

  int print_interval_ = 0;
  
  // Elmore factor for 20-80% slew thresholds.
  static constexpr float elmore_skew_factor_ = 1.39;
  static constexpr int min_print_interval_ = 10;
  static constexpr int max_print_interval_ = 100;

 ///////////
  void repairNet_update(Net* net,
                 const Pin* drvr_pin,
                 Vertex* drvr,
                 bool check_slew,
                 bool check_cap,
                 bool check_fanout,
                 int max_length,  // dbu
                 bool resize_drvr,
                 int& repaired_net_count,
                 int& slew_violations,
                 int& cap_violations,
                 int& fanout_violations,
                 int& length_violations);


  void makeRepeater_update(const char* reason,
                    int x,
                    int y,
                    LibertyCell* buffer_cell,
                    bool resize,
                    int level,
                    // Return values.
                    PinSeq& load_pins,
                    float& repeater_cap,
                    float& repeater_fanout,
                    float& repeater_max_slew,
                    Net*& out_net,
                    Pin*& repeater_in_pin,
                    Pin*& repeater_out_pin,
                    float drvr_output_cap, // new
                    float drvr_output_slew //new
                    ); 
  
  void repairNetWire_update(
    const BufferedNetPtr& bnet,
    int level,
    int& wire_length,  // dbu
    PinSeq& load_pins);

  void repairNetJunc_update(
    const BufferedNetPtr& bnet,
    int level,
    int& wire_length,  // dbu
    PinSeq& load_pins);
  
  void makeRepeater_update(const char* reason,
                    const Point& loc,
                    LibertyCell* buffer_cell,
                    bool resize,
                    int level,
                    // Return values.
                    PinSeq& load_pins,
                    float& repeater_cap,
                    float& repeater_fanout,
                    float& repeater_max_slew,
                    float drvr_output_cap,
                    float drvr_output_slew);

  // We do not consider the max wirelength constraint
  // when we do decide if the net is needed to be repaired.
  float getOutputCap(const Pin* drvr_pin, float& max_cap);
  void getSlew(const Pin* drvr_pin, float& max_cap,
    float &output_slew, float &input_slew);
  void createPreBufferedTree(Net* net, const Pin* drvr_pin, float max_cap);
  void createPostBufferedTree(
    const char* reason,
    Instance* buffer_inst, 
    LibertyCell* buffer_cell,
    int x, int y,
    PinSeq& load_pins,
    float repeater_fanout,
    Pin*& repeater_in_pin,
    Pin*& repeater_out_pin,
    float repeater_input_cap,
    float drvr_output_cap,
    float drvr_output_slew); 

  void testBufferedTree(Pin*& repeater_in_pin,
                        Pin*& repeater_out_pin, 
                        Net*& in_net, 
                        Net*& out_net);//YL

  void countNodes(const BufferedTreePtr& tree, 
                                int& sink_count, 
                                int& buffer_count);

  void printTreeToCSV(const std::vector<BufferedTreePtr>& trees, 
                      const std::string& filename,
                      bool buffered_tree_print_flag);
  bool treeContainsBuffer(const BufferedTreePtr& tree);
  std::pair<bool, bool> treeContainsSpecificNode(const BufferedTreePtr& tree);

  void saveBufferedTrees();
  void saveProbNets(); // YL: for MLBuf integration
  int countNodes(const BufferedTreeNodePtr& node);

  std::string getCurrentTimestamp();
  std::string generateUniqueFilename(const std::string& base_name);

  std::vector<BufferedTreePtr> buffered_trees_pre_;
  std::vector<BufferedTreePtr> buffered_trees_post_;
  std::vector<BufferedTreePtr> buffered_trees_mid_; //YL: track feature updating
  std::vector<BufferedTreePtr> buffered_trees_prob_; //YL: save the problematic net before repair
  // BufferedTreePtr buffered_tree_;
  bool store_buffered_trees_flag_ = false;
  int buffer_order_count_ = 0;
  // Recored the order of the inserted buffers

};

}  // namespace rsz
