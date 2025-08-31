/////////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2022, The Regents of the University of California
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

#include "RepairDesign.hh"

#include <vector>
#include <fstream>
#include <filesystem>

#include "BufferedNet.hh"
#include "db_sta/dbNetwork.hh"
#include "rsz/Resizer.hh"
#include "sta/Corner.hh"
#include "sta/Fuzzy.hh"
#include "sta/Graph.hh"
#include "sta/GraphDelayCalc.hh"
#include "sta/Liberty.hh"
#include "sta/NetworkClass.hh"
#include "sta/PathExpanded.hh"
#include "sta/PathRef.hh"
#include "sta/PathVertex.hh"
#include "sta/PortDirection.hh"
#include "sta/Sdc.hh"
#include "sta/Search.hh"
#include "sta/SearchPred.hh"
#include "sta/Units.hh"
#include <chrono> // YL MLBuf
#include <fstream> //YL MLBuf

namespace rsz {

using std::abs;
using std::max;
using std::min;

using utl::RSZ;

using sta::Clock;
using sta::INF;
using sta::InstancePinIterator;
using sta::NetConnectedPinIterator;
using sta::NetIterator;
using sta::NetPinIterator;
using sta::Port;

RepairDesign::RepairDesign(Resizer* resizer) : resizer_(resizer)
{
}

RepairDesign::~RepairDesign() = default;

void RepairDesign::init()
{
  logger_ = resizer_->logger_;
  dbStaState::init(resizer_->sta_);
  db_network_ = resizer_->db_network_;
  dbu_ = resizer_->dbu_;
  pre_checks_ = new PreChecks(resizer_);
  parasitics_src_ = resizer_->getParasiticsSrc();
}

// Repair long wires, max slew, max capacitance, max fanout violations
// The whole enchilada.
// max_wire_length zero for none (meters)
void RepairDesign::repairDesign(double max_wire_length,
                                double slew_margin,
                                double cap_margin,
                                double buffer_gain,
                                bool verbose)
{
  init();
  int repaired_net_count, slew_violations, cap_violations;
  int fanout_violations, length_violations;
  repairDesign(max_wire_length,
               slew_margin,
               cap_margin,
               buffer_gain,
               verbose,
               repaired_net_count,
               slew_violations,
               cap_violations,
               fanout_violations,
               length_violations);

  if (slew_violations > 0) {
    logger_->info(RSZ, 34, "Found {} slew violations.", slew_violations);
  }
  if (fanout_violations > 0) {
    logger_->info(RSZ, 35, "Found {} fanout violations.", fanout_violations);
  }
  if (cap_violations > 0) {
    logger_->info(RSZ, 36, "Found {} capacitance violations.", cap_violations);
  }
  if (length_violations > 0) {
    logger_->info(RSZ, 37, "Found {} long wires.", length_violations);
  }
  if (inserted_buffer_count_ > 0) {
    logger_->info(RSZ,
                  38,
                  "Inserted {} buffers in {} nets.",
                  inserted_buffer_count_,
                  repaired_net_count);
  }
  if (resize_count_ > 0) {
    logger_->info(RSZ, 39, "Resized {} instances.", resize_count_);
  }
}

void RepairDesign::repairDesign(
    double max_wire_length,  // zero for none (meters)
    double slew_margin,
    double cap_margin,
    double buffer_gain,
    bool verbose,
    int& repaired_net_count,
    int& slew_violations,
    int& cap_violations,
    int& fanout_violations,
    int& length_violations,
    bool store_buffered_trees_flag)
{
  if (store_buffered_trees_flag) {
    // // clean
    // std::string runtime_file = std::getenv("TIMERSZ");
    // std::ofstream ofs(runtime_file, std::ios::trunc);
    // if (!ofs) {
    //     std::cerr << "Failed to open file.\n";
    //     return;
    // }
    // std::cout << "File recordORTime cleared successfully.\n";
    // ofs.close();
    repairDesign_update(max_wire_length,
                        slew_margin,
                        cap_margin,
                        buffer_gain,
                        verbose,
                        repaired_net_count,
                        slew_violations,
                        cap_violations,
                        fanout_violations,
                        length_violations);
    // YL: store buffered trees
    // [Note] YL update 20250419 -- do not save buffered trees !!!
    // saveBufferedTrees();
    // std::cout << "--------------Save buffered trees to the csv file----------------" << std::endl;

    // std::cout << "Buffered Trees Pre Size (#net): " << buffered_trees_pre_.size() << std::endl;
    // std::cout << "Buffered Trees Post Size (#buffered net): " << buffered_trees_post_.size() << std::endl;
    // std::cout << "#buffered net (test): " << repaired_net_count << std::endl;
    // std::cout << "#buffer: " << inserted_buffer_count_ << std::endl;

    // YL: save prob nets here for MLBuf integration
    saveProbNets();
    // // std::string runtime_file = std::getenv("TIMERSZ");
    // std::ofstream file(runtime_file, std::ios::app);
    //   if (file.is_open()) {
    //       file << inserted_buffer_count_ << "\n"; // Append the number and a newline
    //       file.close();
    //       std::cout << "inserted_buffer_count_ appended to file successfully." << std::endl;
    //   } else {
    //       std::cerr << "Unable to open recordORTime.txt" << std::endl;
    //       return; // error code
    //   }
 
    return;
  }
  
  init();
  slew_margin_ = slew_margin;
  cap_margin_ = cap_margin;
  buffer_gain_ = buffer_gain;
  bool gain_buffering = (buffer_gain_ != 0.0);

  slew_violations = 0;
  cap_violations = 0;
  fanout_violations = 0;
  length_violations = 0;
  repaired_net_count = 0;
  inserted_buffer_count_ = 0;
  resize_count_ = 0;
  resizer_->resized_multi_output_insts_.clear();

  // keep track of user annotations so we don't remove them
  std::set<std::pair<Vertex*, int>> slew_user_annotated;

  if (gain_buffering) {
    // If gain-based buffering is enabled, we need to override slews in order to
    // get good required time estimates.
    for (int i = resizer_->level_drvr_vertices_.size() - 1; i >= 0; i--) {
      Vertex* drvr = resizer_->level_drvr_vertices_[i];
      for (auto rf : {RiseFall::rise(), RiseFall::fall()}) {
        if (!drvr->slewAnnotated(rf, min_) && !drvr->slewAnnotated(rf, max_)) {
          sta_->setAnnotatedSlew(drvr,
                                 resizer_->tgt_slew_corner_,
                                 sta::MinMaxAll::all(),
                                 rf->asRiseFallBoth(),
                                 resizer_->tgt_slews_[rf->index()]);
        } else {
          slew_user_annotated.insert(std::make_pair(drvr, rf->index()));
        }
      }
    }
    findBufferSizes();
  }

  sta_->checkSlewLimitPreamble();
  sta_->checkCapacitanceLimitPreamble();
  sta_->checkFanoutLimitPreamble();
  sta_->searchPreamble();
  search_->findAllArrivals();

  resizer_->incrementalParasiticsBegin();
  int print_iteration = 0;
  if (resizer_->level_drvr_vertices_.size() > size_t(5) * max_print_interval_) {
    print_interval_ = max_print_interval_;
  } else {
    print_interval_ = min_print_interval_;
  }
  if (verbose) {
    printProgress(print_iteration, false, false, repaired_net_count);
  }
  int max_length = resizer_->metersToDbu(max_wire_length);
  for (int i = resizer_->level_drvr_vertices_.size() - 1; i >= 0; i--) {
    print_iteration++;
    if (verbose) {
      printProgress(print_iteration, false, false, repaired_net_count);
    }
    Vertex* drvr = resizer_->level_drvr_vertices_[i];
    Pin* drvr_pin = drvr->pin();
    Net* net = network_->isTopLevelPort(drvr_pin)
                   ? network_->net(network_->term(drvr_pin))
                   // hier fix
                   : db_network_->dbToSta(db_network_->flatNet(drvr_pin));
    dbNet* net_db = db_network_->staToDb(net);
    bool debug = (drvr_pin == resizer_->debug_pin_);
    if (debug) {
      logger_->setDebugLevel(RSZ, "repair_net", 3);
    }
    if (gain_buffering) {
      search_->findRequireds(drvr->level() + 1);
    }
    if (net && !resizer_->dontTouch(net) && !net_db->isConnectedByAbutment()
        && !sta_->isClock(drvr_pin)
        // Exclude tie hi/low cells and supply nets.
        && !drvr->isConstant()) {
      repairNet(net,
                drvr_pin,
                drvr,
                true,
                true,
                true,
                max_length,
                true,
                repaired_net_count,
                slew_violations,
                cap_violations,
                fanout_violations,
                length_violations);
    }
    if (debug) {
      logger_->setDebugLevel(RSZ, "repair_net", 0);
    }

    if (gain_buffering) {
      for (auto mm : sta::MinMaxAll::all()->range()) {
        for (auto rf : sta::RiseFallBoth::riseFall()->range()) {
          if (!slew_user_annotated.count(std::make_pair(drvr, rf->index()))) {
            const DcalcAnalysisPt* dcalc_ap
                = resizer_->tgt_slew_corner_->findDcalcAnalysisPt(mm);
            drvr->setSlewAnnotated(false, rf, dcalc_ap->index());
          }
        }
      }
    }
  }
  resizer_->updateParasitics();
  if (verbose) {
    printProgress(print_iteration, true, true, repaired_net_count);
  }
  resizer_->incrementalParasiticsEnd();

  if (inserted_buffer_count_ > 0) {
    resizer_->level_drvr_vertices_valid_ = false;
  }
}

// Repair long wires from clock input pins to clock tree root buffer
// because CTS ignores the issue.
// no max_fanout/max_cap checks.
// Use max_wire_length zero for none (meters)
void RepairDesign::repairClkNets(double max_wire_length)
{
  init();
  slew_margin_ = 0.0;
  cap_margin_ = 0.0;
  buffer_gain_ = 0.0;

  // Need slews to resize inserted buffers.
  sta_->findDelays();

  int slew_violations = 0;
  int cap_violations = 0;
  int fanout_violations = 0;
  int length_violations = 0;
  int repaired_net_count = 0;
  inserted_buffer_count_ = 0;
  resize_count_ = 0;
  resizer_->resized_multi_output_insts_.clear();

  resizer_->incrementalParasiticsBegin();
  int max_length = resizer_->metersToDbu(max_wire_length);
  for (Clock* clk : sdc_->clks()) {
    const PinSet* clk_pins = sta_->pins(clk);
    if (clk_pins) {
      for (const Pin* clk_pin : *clk_pins) {
        Net* net = network_->isTopLevelPort(clk_pin)
                       ? network_->net(network_->term(clk_pin))
                       : network_->net(clk_pin);
        if (net && network_->isDriver(clk_pin)) {
          Vertex* drvr = graph_->pinDrvrVertex(clk_pin);
          // Do not resize clock tree gates.
          repairNet(net,
                    clk_pin,
                    drvr,
                    false,
                    false,
                    false,
                    max_length,
                    false,
                    repaired_net_count,
                    slew_violations,
                    cap_violations,
                    fanout_violations,
                    length_violations);
        }
      }
    }
  }
  resizer_->updateParasitics();
  resizer_->incrementalParasiticsEnd();

  if (length_violations > 0) {
    logger_->info(RSZ, 47, "Found {} long wires.", length_violations);
  }
  if (inserted_buffer_count_ > 0) {
    logger_->info(RSZ,
                  48,
                  "Inserted {} buffers in {} nets.",
                  inserted_buffer_count_,
                  repaired_net_count);
    resizer_->level_drvr_vertices_valid_ = false;
  }
}

// Repair one net (for debugging)
void RepairDesign::repairNet(Net* net,
                             double max_wire_length,  // meters
                             double slew_margin,
                             double cap_margin)
{
  init();
  slew_margin_ = slew_margin;
  cap_margin_ = cap_margin;
  buffer_gain_ = 0.0;

  int slew_violations = 0;
  int cap_violations = 0;
  int fanout_violations = 0;
  int length_violations = 0;
  int repaired_net_count = 0;
  inserted_buffer_count_ = 0;
  resize_count_ = 0;
  resizer_->resized_multi_output_insts_.clear();
  resizer_->buffer_moved_into_core_ = false;

  sta_->checkSlewLimitPreamble();
  sta_->checkCapacitanceLimitPreamble();
  sta_->checkFanoutLimitPreamble();

  resizer_->incrementalParasiticsBegin();
  int max_length = resizer_->metersToDbu(max_wire_length);
  PinSet* drivers = network_->drivers(net);
  if (drivers && !drivers->empty()) {
    PinSet::Iterator drvr_iter(drivers);
    const Pin* drvr_pin = drvr_iter.next();
    Vertex* drvr = graph_->pinDrvrVertex(drvr_pin);
    repairNet(net,
              drvr_pin,
              drvr,
              true,
              true,
              true,
              max_length,
              true,
              repaired_net_count,
              slew_violations,
              cap_violations,
              fanout_violations,
              length_violations);
  }
  resizer_->updateParasitics();
  resizer_->incrementalParasiticsEnd();

  if (slew_violations > 0) {
    logger_->info(RSZ, 51, "Found {} slew violations.", slew_violations);
  }
  if (fanout_violations > 0) {
    logger_->info(RSZ, 52, "Found {} fanout violations.", fanout_violations);
  }
  if (cap_violations > 0) {
    logger_->info(RSZ, 53, "Found {} capacitance violations.", cap_violations);
  }
  if (length_violations > 0) {
    logger_->info(RSZ, 54, "Found {} long wires.", length_violations);
  }
  if (inserted_buffer_count_ > 0) {
    logger_->info(RSZ,
                  55,
                  "Inserted {} buffers in {} nets.",
                  inserted_buffer_count_,
                  repaired_net_count);
    resizer_->level_drvr_vertices_valid_ = false;
  }
  if (resize_count_ > 0) {
    logger_->info(RSZ, 56, "Resized {} instances.", resize_count_);
  }
  if (resize_count_ > 0) {
    logger_->info(RSZ, 57, "Resized {} instances.", resize_count_);
  }
}

bool RepairDesign::getCin(const Pin* drvr_pin, float& cin)
{
  Instance* inst = network_->instance(drvr_pin);
  LibertyCell* cell = network_->libertyCell(inst);
  cin = 0;
  if (!network_->isTopLevelPort(drvr_pin) && cell != nullptr
      && resizer_->isLogicStdCell(inst)) {
    sta::LibertyCellPortIterator port_iter(cell);
    int nports = 0;
    while (port_iter.hasNext()) {
      const LibertyPort* port = port_iter.next();
      if (port->direction() == sta::PortDirection::input()) {
        cin += port->capacitance();
        nports++;
      }
    }
    if (!nports) {
      return false;
    }
    cin /= nports;
    return true;
  }
  return false;
}

static float bufferCin(const LibertyCell* cell)
{
  LibertyPort *a, *y;
  cell->bufferPorts(a, y);
  return a->capacitance();
}

void RepairDesign::findBufferSizes()
{
  buffer_sizes_.clear();
  buffer_sizes_ = resizer_->buffer_cells_;
  std::sort(buffer_sizes_.begin(),
            buffer_sizes_.end(),
            [=](LibertyCell* a, LibertyCell* b) {
              return bufferCin(a) < bufferCin(b);
            });
}

bool RepairDesign::performGainBuffering(Net* net,
                                        const Pin* drvr_pin,
                                        int max_fanout)
{
  struct EnqueuedPin
  {
    Pin* pin;
    PathRef required_path;
    Delay required_delay;
    int level;

    Required required(const StaState* sta) const
    {
      if (required_path.isNull()) {
        return INF;
      }
      return required_path.required(sta) - required_delay;
    }

    std::pair<Required, int> sort_label(const StaState* sta) const
    {
      return std::make_pair(required(sta), -level);
    }

    float capacitance(const Network* network)
    {
      return network->libertyPort(pin)->capacitance();
    }
  };

  class PinRequiredHigher
  {
   private:
    const Network* network_;

   public:
    PinRequiredHigher(const Network* network) : network_(network) {}

    bool operator()(const EnqueuedPin& a, const EnqueuedPin& b) const
    {
      auto la = a.sort_label(network_), lb = b.sort_label(network_);
      if (la > lb) {
        return true;
      }
      if (la < lb) {
        return false;
      }
      return sta::stringLess(network_->pathName(a.pin),
                             network_->pathName(b.pin));
    }
  };

  // Collect all sinks
  std::vector<EnqueuedPin> sinks;

  NetConnectedPinIterator* pin_iter = network_->connectedPinIterator(net);
  while (pin_iter->hasNext()) {
    const Pin* pin = pin_iter->next();
    if (pin != drvr_pin && !network_->isTopLevelPort(pin)
        && network_->direction(pin) == sta::PortDirection::input()
        && network_->libertyPort(pin)) {
      Instance* inst = network_->instance(pin);
      if (!resizer_->dontTouch(inst)) {
        Vertex* vertex = graph_->pinLoadVertex(pin);
        PathRef req_path
            = sta_->vertexWorstSlackPath(vertex, sta::MinMax::max());
        sinks.push_back({const_cast<Pin*>(pin), req_path, 0.0, 0});
      } else {
        logger_->warn(RSZ,
                      143,
                      "dont-touch instance {} ignored when buffering net {}",
                      network_->name(inst),
                      network_->name(net));
      }
    }
  }
  delete pin_iter;

  // Keep track of the vertices at the boundary of the tree so we know where
  // to ask for delays to be recomputed
  std::vector<Vertex*> tree_boundary;

  const float max_buf_load = bufferCin(buffer_sizes_.back()) * buffer_gain_;

  float cin;
  bool repaired_net = false;
  if (getCin(drvr_pin, cin)) {
    float load = 0.0;
    for (auto& sink : sinks) {
      load += sink.capacitance(network_);
    }
    std::sort(sinks.begin(), sinks.end(), PinRequiredHigher(network_));

    // Iterate until we satisfy both the gain condition and max_fanout
    // on drvr_pin
    while (sinks.size() > max_fanout || load > cin * buffer_gain_) {
      float load_acc = 0;
      auto it = sinks.begin();
      for (; it != sinks.end(); it++) {
        if (it - sinks.begin() == max_fanout) {
          break;
        }
        float sink_load = it->capacitance(network_);
        if (load_acc + sink_load > max_buf_load
            // always include at least one load
            && it != sinks.begin()) {
          break;
        }
        load_acc += sink_load;
      }
      auto group_end = it;

      // Find the smallest buffer satisfying the gain condition on
      // its output pin
      auto size = buffer_sizes_.begin();
      for (; size != buffer_sizes_.end() - 1; size++) {
        if (bufferCin(*size) > load_acc / buffer_gain_) {
          break;
        }
      }

      if (bufferCin(*size) >= 0.9f * load_acc) {
        // We are getting dimishing returns on inserting a buffer, stop
        // the algorithm here (we might have been called with a low gain value)
        break;
      }

      Net* new_net = resizer_->makeUniqueNet();
      dbNet* net_db = db_network_->staToDb(net);
      dbNet* new_net_db = db_network_->staToDb(new_net);
      new_net_db->setSigType(net_db->getSigType());

      string buffer_name = resizer_->makeUniqueInstName("gain");
      const Point drvr_loc = db_network_->location(drvr_pin);
      Instance* inst = resizer_->makeBuffer(*size,
                                            buffer_name.c_str(),
                                            // TODO: non-top module handling?
                                            db_network_->topInstance(),
                                            drvr_loc);
      LibertyPort *size_in, *size_out;
      (*size)->bufferPorts(size_in, size_out);
      sta_->connectPin(inst, size_in, net);
      sta_->connectPin(inst, size_out, new_net);

      repaired_net = true;
      inserted_buffer_count_++;

      int max_level = 0;
      for (auto it = sinks.begin(); it != group_end; it++) {
        LibertyPort* sink_port = network_->libertyPort(it->pin);
        Instance* sink_inst = network_->instance(it->pin);
        load -= sink_port->capacitance();
        if (it->level > max_level) {
          max_level = it->level;
        }
        sta_->disconnectPin(it->pin);
        sta_->connectPin(sink_inst, sink_port, new_net);
        if (it->level == 0) {
          Pin* new_pin = network_->findPin(sink_inst, sink_port);
          tree_boundary.push_back(graph_->pinLoadVertex(new_pin));
        }
      }

      Pin* new_input_pin = network_->findPin(inst, size_in);

      Delay buffer_delay = resizer_->bufferDelay(
          *size, load_acc, resizer_->tgt_slew_dcalc_ap_);

      auto new_pin = EnqueuedPin{new_input_pin,
                                 (group_end - 1)->required_path,
                                 (group_end - 1)->required_delay + buffer_delay,
                                 max_level + 1};

      sinks.erase(sinks.begin(), group_end);
      sinks.insert(
          std::upper_bound(
              sinks.begin(), sinks.end(), new_pin, PinRequiredHigher(network_)),
          new_pin);

      load += size_in->capacitance();
    }
  }

  sta_->ensureLevelized();
  sta::Level max_level = 0;
  for (auto vertex : tree_boundary) {
    max_level = std::max(vertex->level(), max_level);
  }
  sta_->findDelays(max_level);
  search_->findArrivals(max_level);

  return repaired_net;
}

void RepairDesign::repairNet(Net* net,
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
                             int& length_violations)
{
  // Hands off special nets.
  if (!db_network_->isSpecial(net)) {
    debugPrint(logger_,
               RSZ,
               "repair_net",
               1,
               "repair net {}",
               sdc_network_->pathName(drvr_pin));
    const Corner* corner = sta_->cmdCorner();
    bool repaired_net = false;

    if (buffer_gain_ != 0.0) {
      float fanout, max_fanout, fanout_slack;
      sta_->checkFanout(drvr_pin, max_, fanout, max_fanout, fanout_slack);

      int resized = resizer_->resizeToTargetSlew(drvr_pin);
      if (performGainBuffering(net, drvr_pin, max_fanout)) {
        repaired_net = true;
      }
      // Resize again post buffering as the load changed
      resized += resizer_->resizeToTargetSlew(drvr_pin);
      if (resized > 0) {
        repaired_net = true;
        resize_count_ += 1;
      }
    }

    if (check_fanout) {
      float fanout, max_fanout, fanout_slack;
      sta_->checkFanout(drvr_pin, max_, fanout, max_fanout, fanout_slack);
      if (max_fanout > 0.0 && fanout_slack < 0.0) {
        fanout_violations++;
        repaired_net = true;

        debugPrint(logger_, RSZ, "repair_net", 3, "fanout violation");
        LoadRegion region = findLoadRegions(drvr_pin, max_fanout);
        corner_ = corner;
        makeRegionRepeaters(region,
                            max_fanout,
                            1,
                            drvr_pin,
                            check_slew,
                            check_cap,
                            max_length,
                            resize_drvr);
      }
    }

    // Resize the driver to normalize slews before repairing limit violations.
    if (parasitics_src_ == ParasiticsSrc::placement && resize_drvr) {
      resize_count_ += resizer_->resizeToTargetSlew(drvr_pin);
    }
    // For tristate nets all we can do is resize the driver.
    if (!resizer_->isTristateDriver(drvr_pin)) {
      BufferedNetPtr bnet = resizer_->makeBufferedNetSteiner(drvr_pin, corner);
      if (bnet) {
        resizer_->ensureWireParasitic(drvr_pin, net);
        graph_delay_calc_->findDelays(drvr);

        float max_cap = INF;
        int wire_length = bnet->maxLoadWireLength();
        bool need_repair = needRepair(drvr_pin,
                                      corner,
                                      max_length,
                                      wire_length,
                                      check_cap,
                                      check_slew,
                                      max_cap,
                                      slew_violations,
                                      cap_violations,
                                      length_violations);

        if (need_repair) {
          if (parasitics_src_ == ParasiticsSrc::global_routing && resize_drvr) {
            resize_count_ += resizer_->resizeToTargetSlew(drvr_pin);
            wire_length = bnet->maxLoadWireLength();
            need_repair = needRepair(drvr_pin,
                                     corner,
                                     max_length,
                                     wire_length,
                                     check_cap,
                                     check_slew,
                                     max_cap,
                                     slew_violations,
                                     cap_violations,
                                     length_violations);
          }
          if (need_repair) {
            Point drvr_loc = db_network_->location(drvr->pin());
            debugPrint(
                logger_,
                RSZ,
                "repair_net",
                1,
                "driver {} ({} {}) l={}",
                sdc_network_->pathName(drvr_pin),
                units_->distanceUnit()->asString(dbuToMeters(drvr_loc.getX()),
                                                 1),
                units_->distanceUnit()->asString(dbuToMeters(drvr_loc.getY()),
                                                 1),
                units_->distanceUnit()->asString(dbuToMeters(wire_length), 1));
            repairNet(bnet, drvr_pin, max_cap, max_length, corner);
            repaired_net = true;

            if (resize_drvr) {
              resize_count_ += resizer_->resizeToTargetSlew(drvr_pin);
            }
          }
        }
      }
    }
    if (repaired_net) {
      repaired_net_count++;
    }
  }
}

bool RepairDesign::needRepairSlew(const Pin* drvr_pin,
                                  int& slew_violations,
                                  float& max_cap,
                                  const Corner*& corner)
{
  bool repair_slew = false;
  float slew1, slew_slack1, max_slew1;
  const Corner* corner1;
  // Check slew at the driver.
  checkSlew(drvr_pin, slew1, max_slew1, slew_slack1, corner1);
  // Max slew violations at the driver pin are repaired by reducing the
  // load capacitance. Wire resistance may shield capacitance from the
  // driver but so this is conservative.
  // Find max load cap that corresponds to max_slew.
  LibertyPort* drvr_port = network_->libertyPort(drvr_pin);
  if (corner1 && max_slew1 > 0.0) {
    if (drvr_port) {
      float max_cap1 = findSlewLoadCap(drvr_port, max_slew1, corner1);
      max_cap = min(max_cap, max_cap1);
    }
    corner = corner1;
    if (slew_slack1 < 0.0) {
      debugPrint(logger_,
                 RSZ,
                 "repair_net",
                 2,
                 "drvr slew violation slew={} max_slew={}",
                 delayAsString(slew1, this, 3),
                 delayAsString(max_slew1, this, 3));
      repair_slew = true;
      slew_violations++;
    }
  }
  // Check slew at the loads.
  // Note that many liberty libraries do not have max_transition attributes on
  // input pins.
  // Max slew violations at the load pins are repaired by inserting buffers
  // and reducing the wire length to the load.
  resizer_->checkLoadSlews(
      drvr_pin, slew_margin_, slew1, max_slew1, slew_slack1, corner1);
  if (slew_slack1 < 0.0) {
    debugPrint(logger_,
               RSZ,
               "repair_net",
               2,
               "load slew violation load_slew={} max_slew={}",
               delayAsString(slew1, this, 3),
               delayAsString(max_slew1, this, 3));
    corner = corner1;
    // Don't double count the driver/load on same net.
    if (!repair_slew) {
      slew_violations++;
    }
    repair_slew = true;
  }

  return repair_slew;
}

bool RepairDesign::needRepairCap(const Pin* drvr_pin,
                                 int& cap_violations,
                                 float& max_cap,
                                 const Corner*& corner)
{
  float cap1, max_cap1, cap_slack1;
  const Corner* corner1;
  const RiseFall* tr1;
  sta_->checkCapacitance(
      drvr_pin, nullptr, max_, corner1, tr1, cap1, max_cap1, cap_slack1);
  if (max_cap1 > 0.0 && corner1) {
    max_cap1 *= (1.0 - cap_margin_ / 100.0);
    max_cap = max_cap1;
    if (cap1 > max_cap1) {
      corner = corner1;
      cap_violations++;
      return true;
    }
  }

  return false;
}

bool RepairDesign::needRepairWire(const int max_length,
                                  const int wire_length,
                                  int& length_violations)
{
  if (max_length && wire_length > max_length) {
    length_violations++;
    return true;
  }
  return false;
}

bool RepairDesign::needRepair(const Pin* drvr_pin,
                              const Corner*& corner,
                              const int max_length,
                              const int wire_length,
                              const bool check_cap,
                              const bool check_slew,
                              float& max_cap,
                              int& slew_violations,
                              int& cap_violations,
                              int& length_violations)
{
  bool repair_cap = false;
  bool repair_slew = false;
  if (check_cap) {
    repair_cap = needRepairCap(drvr_pin, cap_violations, max_cap, corner);
  }
  bool repair_wire = needRepairWire(max_length, wire_length, length_violations);
  if (check_slew) {
    repair_slew = needRepairSlew(drvr_pin, slew_violations, max_cap, corner);
  }

  return repair_cap || repair_wire || repair_slew;
}

bool RepairDesign::checkLimits(const Pin* drvr_pin,
                               bool check_slew,
                               bool check_cap,
                               bool check_fanout)
{
  if (check_cap) {
    float cap1, max_cap1, cap_slack1;
    const Corner* corner1;
    const RiseFall* tr1;
    sta_->checkCapacitance(
        drvr_pin, nullptr, max_, corner1, tr1, cap1, max_cap1, cap_slack1);
    max_cap1 *= (1.0 - cap_margin_ / 100.0);
    if (cap1 < max_cap1) {
      return true;
    }
  }
  if (check_fanout) {
    float fanout, fanout_slack, max_fanout;
    sta_->checkFanout(drvr_pin, max_, fanout, max_fanout, fanout_slack);
    if (fanout_slack < 0.0) {
      return true;
    }
  }
  if (check_slew) {
    float slew1, slew_slack1, max_slew1;
    const Corner* corner1;
    checkSlew(drvr_pin, slew1, max_slew1, slew_slack1, corner1);
    if (slew_slack1 < 0.0) {
      return true;
    }
    resizer_->checkLoadSlews(
        drvr_pin, slew_margin_, slew1, max_slew1, slew_slack1, corner1);
    if (slew_slack1 < 0.0) {
      return true;
    }
  }
  return false;
}

void RepairDesign::checkSlew(const Pin* drvr_pin,
                             // Return values.
                             Slew& slew,
                             float& limit,
                             float& slack,
                             const Corner*& corner)
{
  slack = INF;
  limit = INF;
  corner = nullptr;

  const Corner* corner1;
  const RiseFall* tr1;
  Slew slew1;
  float limit1, slack1;
  sta_->checkSlew(
      drvr_pin, nullptr, max_, false, corner1, tr1, slew1, limit1, slack1);
  if (corner1) {
    limit1 *= (1.0 - slew_margin_ / 100.0);
    slack1 = limit1 - slew1;
    if (slack1 < slack) {
      slew = slew1;
      limit = limit1;
      slack = slack1;
      corner = corner1;
    }
  }
}

float RepairDesign::bufferInputMaxSlew(LibertyCell* buffer,
                                       const Corner* corner) const
{
  LibertyPort *input, *output;
  buffer->bufferPorts(input, output);
  return resizer_->maxInputSlew(input, corner);
}

// Find the output port load capacitance that results in slew.
double RepairDesign::findSlewLoadCap(LibertyPort* drvr_port,
                                     double slew,
                                     const Corner* corner)
{
  const DcalcAnalysisPt* dcalc_ap = corner->findDcalcAnalysisPt(max_);
  double drvr_res = drvr_port->driveResistance();
  if (drvr_res == 0.0) {
    return INF;
  }
  // cap1 lower bound
  // cap2 upper bound
  double cap1 = 0.0;
  double cap2 = slew / drvr_res * 2;
  double tol = .01;  // 1%
  double diff1 = gateSlewDiff(drvr_port, cap2, slew, dcalc_ap);
  // binary search for diff = 0.
  while (abs(cap1 - cap2) > max(cap1, cap2) * tol) {
    if (diff1 < 0.0) {
      cap1 = cap2;
      cap2 *= 2;
      diff1 = gateSlewDiff(drvr_port, cap2, slew, dcalc_ap);
    } else {
      double cap3 = (cap1 + cap2) / 2.0;
      double diff2 = gateSlewDiff(drvr_port, cap3, slew, dcalc_ap);
      if (diff2 < 0.0) {
        cap1 = cap3;
      } else {
        cap2 = cap3;
        diff1 = diff2;
      }
    }
  }
  return cap1;
}

// objective function
double RepairDesign::gateSlewDiff(LibertyPort* drvr_port,
                                  double load_cap,
                                  double slew,
                                  const DcalcAnalysisPt* dcalc_ap)
{
  ArcDelay delays[RiseFall::index_count];
  Slew slews[RiseFall::index_count];
  resizer_->gateDelays(drvr_port, load_cap, dcalc_ap, delays, slews);
  Slew gate_slew
      = max(slews[RiseFall::riseIndex()], slews[RiseFall::fallIndex()]);
  return gate_slew - slew;
}

////////////////////////////////////////////////////////////////

void RepairDesign::repairNet(const BufferedNetPtr& bnet,
                             const Pin* drvr_pin,
                             float max_cap,
                             int max_length,  // dbu
                             const Corner* corner)
{
  drvr_pin_ = drvr_pin;
  max_cap_ = max_cap;
  max_length_ = max_length;
  corner_ = corner;

  int wire_length;
  PinSeq load_pins;
  repairNet(bnet, 0, wire_length, load_pins);
}

void RepairDesign::repairNet(const BufferedNetPtr& bnet,
                             int level,
                             // Return values.
                             // Remaining parasiics after repeater insertion.
                             int& wire_length,  // dbu
                             PinSeq& load_pins)
{
  switch (bnet->type()) {
    case BufferedNetType::wire:
      repairNetWire(bnet, level, wire_length, load_pins);
      break;
    case BufferedNetType::junction:
      repairNetJunc(bnet, level, wire_length, load_pins);
      break;
    case BufferedNetType::load:
      repairNetLoad(bnet, level, wire_length, load_pins);
      break;
    case BufferedNetType::buffer:
      logger_->critical(RSZ, 72, "unhandled BufferedNet type");
      break;
  }
}

void RepairDesign::repairNetWire(
    const BufferedNetPtr& bnet,
    int level,
    // Return values.
    // Remaining parasiics after repeater insertion.
    int& wire_length,  // dbu
    PinSeq& load_pins)
{
    if(store_buffered_trees_flag_==true)
  {
    repairNetWire_update(bnet, level, wire_length, load_pins);
    return;
  }

  debugPrint(logger_,
             RSZ,
             "repair_net",
             3,
             "{:{}s}{}",
             "",
             level,
             bnet->to_string(resizer_));
  int wire_length_ref;
  repairNet(bnet->ref(), level + 1, wire_length_ref, load_pins);
  float max_load_slew = bnet->maxLoadSlew();
  float max_load_slew_margined
      = max_load_slew;  // maxSlewMargined(max_load_slew);

  Point to_loc = bnet->ref()->location();
  int to_x = to_loc.getX();
  int to_y = to_loc.getY();
  Point from_loc = bnet->location();
  int length = Point::manhattanDistance(from_loc, to_loc);
  wire_length = wire_length_ref + length;
  int from_x = from_loc.getX();
  int from_y = from_loc.getY();
  debugPrint(logger_,
             RSZ,
             "repair_net",
             3,
             "{:{}s}wl={} l={}",
             "",
             level,
             units_->distanceUnit()->asString(dbuToMeters(wire_length), 1),
             units_->distanceUnit()->asString(dbuToMeters(length), 1));
  double length1 = dbuToMeters(length);
  double wire_res, wire_cap;
  bnet->wireRC(corner_, resizer_, wire_res, wire_cap);
  // ref_cap includes ref's wire cap
  double ref_cap = bnet->ref()->cap();
  double load_cap = length1 * wire_cap + ref_cap;

  float r_drvr = resizer_->driveResistance(drvr_pin_);
  double load_slew = (r_drvr + dbuToMeters(wire_length) * wire_res) * load_cap
                     * elmore_skew_factor_;
  debugPrint(logger_,
             RSZ,
             "repair_net",
             3,
             "{:{}s}load_slew={} r_drvr={}",
             "",
             level,
             delayAsString(load_slew, this, 3),
             units_->resistanceUnit()->asString(r_drvr, 3));

  LibertyCell* buffer_cell = resizer_->findTargetCell(
      resizer_->buffer_lowest_drive_, load_cap, false);
  bnet->setCapacitance(load_cap);
  bnet->setFanout(bnet->ref()->fanout());

  // Check that the slew limit specified is within the bounds of reason.
  pre_checks_->checkSlewLimit(ref_cap, max_load_slew);
  //============================================================================
  // Back up from pt to from_pt adding repeaters as necessary for
  // length/max_cap/max_slew violations.
  while ((max_length_ > 0 && wire_length > max_length_)
         || (wire_cap > 0.0 && max_cap_ > 0.0 && load_cap > max_cap_)
         || load_slew > max_load_slew_margined) {
    // Make the wire a bit shorter than necessary to allow for
    // offset from instance origin to pin and detailed placement movement.
    constexpr double length_margin = .05;
    bool split_wire = false;
    bool resize = true;
    // Distance from repeater to ref_.
    //              length
    // from----------------------------to/ref
    //
    //                     split_length
    // from-------repeater-------------to/ref
    int split_length = std::numeric_limits<int>::max();
    if (max_length_ > 0 && wire_length > max_length_) {
      debugPrint(logger_,
                 RSZ,
                 "repair_net",
                 3,
                 "{:{}s}max wire length violation {} > {}",
                 "",
                 level,
                 units_->distanceUnit()->asString(dbuToMeters(wire_length), 1),
                 units_->distanceUnit()->asString(dbuToMeters(max_length_), 1));
      split_length = min(max(max_length_ - wire_length_ref, 0), length / 2);

      split_wire = true;
    }
    if (wire_cap > 0.0 && load_cap > max_cap_) {
      debugPrint(logger_,
                 RSZ,
                 "repair_net",
                 3,
                 "{:{}s}max cap violation {} > {}",
                 "",
                 level,
                 units_->capacitanceUnit()->asString(load_cap, 3),
                 units_->capacitanceUnit()->asString(max_cap_, 3));
      if (ref_cap > max_cap_) {
        split_length = 0;
        split_wire = false;
      } else {
        split_length
            = min(split_length, metersToDbu((max_cap_ - ref_cap) / wire_cap));
        split_wire = true;
      }
    }
    if (load_slew > max_load_slew_margined) {
      debugPrint(logger_,
                 RSZ,
                 "repair_net",
                 3,
                 "{:{}s}max load slew violation {} > {}",
                 "",
                 level,
                 delayAsString(load_slew, this, 3),
                 delayAsString(max_load_slew_margined, this, 3));
      // Using elmore delay to approximate wire
      // load_slew = (Rbuffer + (L+Lref)*Rwire) * (L*Cwire + Cref) *
      // elmore_skew_factor_ Setting this to max_load_slew_margined is a
      // quadratic in L L^2*Rwire*Cwire + L*(Rbuffer*Cwire + Rwire*Cref +
      // Rwire*Cwire*Lref)
      //   + Rbuffer*Cref - max_load_slew_margined/elmore_skew_factor_
      // Solve using quadradic eqn for L.
      float wire_length_ref1 = dbuToMeters(wire_length_ref);
      float r_buffer = resizer_->bufferDriveResistance(buffer_cell);
      float a = wire_res * wire_cap;
      float b = r_buffer * wire_cap + wire_res * ref_cap
                + wire_res * wire_cap * wire_length_ref1;
      float c = r_buffer * ref_cap + wire_length_ref1 * wire_res * ref_cap
                - max_load_slew_margined / elmore_skew_factor_;
      float l = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
      if (l >= 0.0) {
        if (split_length > 0.0) {
          split_length = min(split_length, metersToDbu(l));
        } else {
          split_length = metersToDbu(l);
        }
        split_wire = true;
        resize = false;
      } else {
        split_length = 0;
        split_wire = false;
        resize = true;
      }
    }

    if (split_wire) {
      debugPrint(
          logger_,
          RSZ,
          "repair_net",
          3,
          "{:{}s}split length={}",
          "",
          level,
          units_->distanceUnit()->asString(dbuToMeters(split_length), 1));
      // Distance from to_pt to repeater backward toward from_pt.
      // Note that split_length can be longer than the wire length
      // because it is the maximum value that satisfies max slew/cap.
      double buf_dist = (split_length >= length)
                            ? length
                            : split_length * (1.0 - length_margin);
      double dx = from_x - to_x;
      double dy = from_y - to_y;
      double d = (length == 0) ? 0.0 : buf_dist / length;
      int buf_x = to_x + d * dx;
      int buf_y = to_y + d * dy;
      float repeater_cap, repeater_fanout;
      makeRepeater("wire",
                   Point(buf_x, buf_y),
                   buffer_cell,
                   resize,
                   level,
                   load_pins,
                   repeater_cap,
                   repeater_fanout,
                   max_load_slew);
      // Update for the next round.
      length -= buf_dist;
      wire_length = length;
      to_x = buf_x;
      to_y = buf_y;

      length1 = dbuToMeters(length);
      wire_length_ref = 0.0;
      load_cap = repeater_cap + length1 * wire_cap;
      ref_cap = repeater_cap;
      max_load_slew_margined
          = max_load_slew;  // maxSlewMargined(max_load_slew);
      load_slew
          = (r_drvr + length1 * wire_res) * load_cap * elmore_skew_factor_;
      buffer_cell = resizer_->findTargetCell(
          resizer_->buffer_lowest_drive_, load_cap, false);

      bnet->setCapacitance(load_cap);
      bnet->setFanout(repeater_fanout);
      bnet->setMaxLoadSlew(max_load_slew);

      debugPrint(logger_,
                 RSZ,
                 "repair_net",
                 3,
                 "{:{}s}l={}",
                 "",
                 level,
                 units_->distanceUnit()->asString(length1, 1));
    } else {
      break;
    }
  }
}

float RepairDesign::maxSlewMargined(float max_slew)
{
  return max_slew * (1.0 - slew_margin_ / 100.0);
}

void RepairDesign::repairNetJunc(
    const BufferedNetPtr& bnet,
    int level,
    // Return values.
    // Remaining parasiics after repeater insertion.
    int& wire_length,  // dbu
    PinSeq& load_pins)
{
  if(store_buffered_trees_flag_==true)
  {
    repairNetJunc_update(bnet, level, wire_length, load_pins);
    return;
  }

  debugPrint(logger_,
             RSZ,
             "repair_net",
             3,
             "{:{}s}{}",
             "",
             level,
             bnet->to_string(resizer_));
  Point loc = bnet->location();
  double wire_res, wire_cap;
  resizer_->wireSignalRC(corner_, wire_res, wire_cap);

  BufferedNetPtr left = bnet->ref();
  int wire_length_left = 0;
  PinSeq loads_left;
  repairNet(left, level + 1, wire_length_left, loads_left);
  float cap_left = left->cap();
  float fanout_left = left->fanout();
  float max_load_slew_left = left->maxLoadSlew();

  BufferedNetPtr right = bnet->ref2();
  int wire_length_right = 0;
  PinSeq loads_right;
  repairNet(right, level + 1, wire_length_right, loads_right);
  float cap_right = right->cap();
  float fanout_right = right->fanout();
  float max_load_slew_right = right->maxLoadSlew();

  debugPrint(logger_,
             RSZ,
             "repair_net",
             3,
             "{:{}s}left  l={} cap={} fanout={}",
             "",
             level,
             units_->distanceUnit()->asString(dbuToMeters(wire_length_left), 1),
             units_->capacitanceUnit()->asString(cap_left, 3),
             fanout_left);
  debugPrint(
      logger_,
      RSZ,
      "repair_net",
      3,
      "{:{}s}right l={} cap={} fanout={}",
      "",
      level,
      units_->distanceUnit()->asString(dbuToMeters(wire_length_right), 1),
      units_->capacitanceUnit()->asString(cap_right, 3),
      fanout_right);

  wire_length = wire_length_left + wire_length_right;
  float load_cap = cap_left + cap_right;
  float max_load_slew = min(max_load_slew_left, max_load_slew_right);
  float max_load_slew_margined
      = max_load_slew;  // maxSlewMargined(max_load_slew);
  LibertyCell* buffer_cell = resizer_->findTargetCell(
      resizer_->buffer_lowest_drive_, load_cap, false);

  // Check for violations when the left/right branches are combined.
  // Add a buffer to left or right branch to stay under the max
  // cap/length/fanout.
  bool repeater_left = false;
  bool repeater_right = false;
  float r_drvr = resizer_->driveResistance(drvr_pin_);
  float load_slew = (r_drvr + dbuToMeters(wire_length) * wire_res) * load_cap
                    * elmore_skew_factor_;
  bool load_slew_violation = load_slew > max_load_slew_margined;
  const char* repeater_reason = nullptr;
  // Driver slew checks were converted to max cap.
  if (load_slew_violation) {
    debugPrint(logger_,
               RSZ,
               "repair_net",
               3,
               "{:{}s}load slew violation {} > {}",
               "",
               level,
               delayAsString(load_slew, this, 3),
               delayAsString(max_load_slew_margined, this, 3));
    float slew_left = (r_drvr + dbuToMeters(wire_length_left) * wire_res)
                      * cap_left * elmore_skew_factor_;
    float slew_slack_left = maxSlewMargined(max_load_slew_left) - slew_left;
    float slew_right = (r_drvr + dbuToMeters(wire_length_right) * wire_res)
                       * cap_right * elmore_skew_factor_;
    float slew_slack_right = maxSlewMargined(max_load_slew_right) - slew_right;
    debugPrint(logger_,
               RSZ,
               "repair_net",
               3,
               "{:{}s} slew slack left={} right={}",
               "",
               level,
               delayAsString(slew_slack_left, this, 3),
               delayAsString(slew_slack_right, this, 3));
    // Isolate the branch with the smaller slack.
    if (slew_slack_left < slew_slack_right) {
      repeater_left = true;
    } else {
      repeater_right = true;
    }
    repeater_reason = "load_slew";
  }
  bool cap_violation = (cap_left + cap_right) > max_cap_;
  if (cap_violation) {
    debugPrint(logger_, RSZ, "repair_net", 3, "{:{}s}cap violation", "", level);
    if (cap_left > cap_right) {
      repeater_left = true;
    } else {
      repeater_right = true;
    }
    repeater_reason = "max_cap";
  }
  bool length_violation
      = max_length_ > 0 && (wire_length_left + wire_length_right) > max_length_;
  if (length_violation) {
    debugPrint(
        logger_, RSZ, "repair_net", 3, "{:{}s}length violation", "", level);
    if (wire_length_left > wire_length_right) {
      repeater_left = true;
    } else {
      repeater_right = true;
    }
    repeater_reason = "max_length";
  }

  if (repeater_left) {
    makeRepeater(repeater_reason,
                 loc,
                 buffer_cell,
                 true,
                 level,
                 loads_left,
                 cap_left,
                 fanout_left,
                 max_load_slew_left);
    wire_length_left = 0;
  }
  if (repeater_right) {
    makeRepeater(repeater_reason,
                 loc,
                 buffer_cell,
                 true,
                 level,
                 loads_right,
                 cap_right,
                 fanout_right,
                 max_load_slew_right);
    wire_length_right = 0;
  }

  // Update after left/right repeaters are inserted.
  wire_length = wire_length_left + wire_length_right;

  bnet->setCapacitance(cap_left + cap_right);
  bnet->setFanout(fanout_right + fanout_left);
  bnet->setMaxLoadSlew(min(max_load_slew_left, max_load_slew_right));

  // Union left/right load pins.
  for (const Pin* load_pin : loads_left) {
    load_pins.push_back(load_pin);
  }
  for (const Pin* load_pin : loads_right) {
    load_pins.push_back(load_pin);
  }
}

void RepairDesign::repairNetLoad(
    const BufferedNetPtr& bnet,
    int level,
    // Return values.
    // Remaining parasiics after repeater insertion.
    int& wire_length,  // dbu
    PinSeq& load_pins)
{
  debugPrint(logger_,
             RSZ,
             "repair_net",
             3,
             "{:{}s}{}",
             "",
             level,
             bnet->to_string(resizer_));
  const Pin* load_pin = bnet->loadPin();
  debugPrint(logger_,
             RSZ,
             "repair_net",
             2,
             "{:{}s}load {}",
             "",
             level,
             sdc_network_->pathName(load_pin));
  wire_length = 0;
  load_pins.push_back(load_pin);
}

////////////////////////////////////////////////////////////////

LoadRegion::LoadRegion() = default;

LoadRegion::LoadRegion(PinSeq& pins, Rect& bbox) : pins_(pins), bbox_(bbox)
{
}

LoadRegion RepairDesign::findLoadRegions(const Pin* drvr_pin, int max_fanout)
{
  PinSeq loads = findLoads(drvr_pin);
  Rect bbox = findBbox(loads);
  LoadRegion region(loads, bbox);
  subdivideRegion(region, max_fanout);
  return region;
}

void RepairDesign::subdivideRegion(LoadRegion& region, int max_fanout)
{
  if (region.pins_.size() > max_fanout && region.bbox_.dx() > dbu_
      && region.bbox_.dy() > dbu_) {
    int x_min = region.bbox_.xMin();
    int x_max = region.bbox_.xMax();
    int y_min = region.bbox_.yMin();
    int y_max = region.bbox_.yMax();
    region.regions_.resize(2);
    int64_t x_mid = (x_min + x_max) / 2;
    int64_t y_mid = (y_min + y_max) / 2;
    bool horz_partition;
    if (region.bbox_.dx() > region.bbox_.dy()) {
      region.regions_[0].bbox_ = Rect(x_min, y_min, x_mid, y_max);
      region.regions_[1].bbox_ = Rect(x_mid, y_min, x_max, y_max);
      horz_partition = true;
    } else {
      region.regions_[0].bbox_ = Rect(x_min, y_min, x_max, y_mid);
      region.regions_[1].bbox_ = Rect(x_min, y_mid, x_max, y_max);
      horz_partition = false;
    }
    for (const Pin* pin : region.pins_) {
      Point loc = db_network_->location(pin);
      int x = loc.x();
      int y = loc.y();
      if ((horz_partition && x <= x_mid) || (!horz_partition && y <= y_mid)) {
        region.regions_[0].pins_.push_back(pin);
      } else if ((horz_partition && x > x_mid)
                 || (!horz_partition && y > y_mid)) {
        region.regions_[1].pins_.push_back(pin);
      } else {
        logger_->critical(RSZ, 83, "pin outside regions");
      }
    }
    region.pins_.clear();
    for (LoadRegion& sub : region.regions_) {
      if (!sub.pins_.empty()) {
        subdivideRegion(sub, max_fanout);
      }
    }
  }
}

void RepairDesign::makeRegionRepeaters(LoadRegion& region,
                                       int max_fanout,
                                       int level,
                                       const Pin* drvr_pin,
                                       bool check_slew,
                                       bool check_cap,
                                       int max_length,
                                       bool resize_drvr)
{
  // Leaf regions have less than max_fanout pins and are buffered
  // by the enclosing region.
  if (!region.regions_.empty()) {
    // Buffer from the bottom up.
    for (LoadRegion& sub : region.regions_) {
      makeRegionRepeaters(sub,
                          max_fanout,
                          level + 1,
                          drvr_pin,
                          check_slew,
                          check_cap,
                          max_length,
                          resize_drvr);
    }

    PinSeq repeater_inputs;
    PinSeq repeater_loads;
    for (LoadRegion& sub : region.regions_) {
      PinSeq& sub_pins = sub.pins_;
      while (!sub_pins.empty()) {
        repeater_loads.push_back(sub_pins.back());
        sub_pins.pop_back();
        if (repeater_loads.size() == max_fanout) {
          makeFanoutRepeater(repeater_loads,
                             repeater_inputs,
                             region.bbox_,
                             findClosedPinLoc(drvr_pin, repeater_loads),
                             check_slew,
                             check_cap,
                             max_length,
                             resize_drvr);
        }
      }
    }
    while (!region.pins_.empty()) {
      repeater_loads.push_back(region.pins_.back());
      region.pins_.pop_back();
      if (repeater_loads.size() == max_fanout) {
        makeFanoutRepeater(repeater_loads,
                           repeater_inputs,
                           region.bbox_,
                           findClosedPinLoc(drvr_pin, repeater_loads),
                           check_slew,
                           check_cap,
                           max_length,
                           resize_drvr);
      }
    }
    if (!repeater_loads.empty() && repeater_loads.size() >= max_fanout / 2) {
      makeFanoutRepeater(repeater_loads,
                         repeater_inputs,
                         region.bbox_,
                         findClosedPinLoc(drvr_pin, repeater_loads),
                         check_slew,
                         check_cap,
                         max_length,
                         resize_drvr);
    } else {
      region.pins_ = std::move(repeater_loads);
    }

    for (const Pin* pin : repeater_inputs) {
      region.pins_.push_back(pin);
    }
  }
}

void RepairDesign::makeFanoutRepeater(PinSeq& repeater_loads,
                                      PinSeq& repeater_inputs,
                                      const Rect& bbox,
                                      const Point& loc,
                                      bool check_slew,
                                      bool check_cap,
                                      int max_length,
                                      bool resize_drvr)
{
  float ignore2, ignore3, ignore4;
  Net* out_net;
  Pin *repeater_in_pin, *repeater_out_pin;
  makeRepeater("fanout",
               loc.x(),
               loc.y(),
               resizer_->buffer_lowest_drive_,
               false,
               1,
               repeater_loads,
               ignore2,
               ignore3,
               ignore4,
               out_net,
               repeater_in_pin,
               repeater_out_pin);
  Vertex* repeater_out_vertex = graph_->pinDrvrVertex(repeater_out_pin);
  int repaired_net_count, slew_violations, cap_violations = 0;
  int fanout_violations, length_violations = 0;
  repairNet(out_net,
            repeater_out_pin,
            repeater_out_vertex,
            check_slew,
            check_cap,
            false /* check_fanout */,
            max_length,
            resize_drvr,
            repaired_net_count,
            slew_violations,
            cap_violations,
            fanout_violations,
            length_violations);
  repeater_inputs.push_back(repeater_in_pin);
  repeater_loads.clear();
}

Rect RepairDesign::findBbox(PinSeq& pins)
{
  Rect bbox;
  bbox.mergeInit();
  for (const Pin* pin : pins) {
    Point loc = db_network_->location(pin);
    Rect r(loc.x(), loc.y(), loc.x(), loc.y());
    bbox.merge(r);
  }
  return bbox;
}

PinSeq RepairDesign::findLoads(const Pin* drvr_pin)
{
  PinSeq loads;
  Pin* drvr_pin1 = const_cast<Pin*>(drvr_pin);
  PinSeq drvrs;
  PinSet visited_drvrs(db_network_);
  sta::FindNetDrvrLoads visitor(
      drvr_pin1, visited_drvrs, loads, drvrs, network_);
  network_->visitConnectedPins(drvr_pin1, visitor);
  return loads;
}

Point RepairDesign::findClosedPinLoc(const Pin* drvr_pin, PinSeq& pins)
{
  Point drvr_loc = db_network_->location(drvr_pin);
  Point closest_pt = drvr_loc;
  int64_t closest_dist = std::numeric_limits<int64_t>::max();
  for (const Pin* pin : pins) {
    Point loc = db_network_->location(pin);
    int64_t dist = Point::manhattanDistance(loc, drvr_loc);
    if (dist < closest_dist) {
      closest_pt = loc;
      closest_dist = dist;
    }
  }
  return closest_pt;
}

bool RepairDesign::isRepeater(const Pin* load_pin)
{
  dbInst* db_inst = db_network_->staToDb(network_->instance(load_pin));
  odb::dbSourceType source = db_inst->getSourceType();
  return source == odb::dbSourceType::TIMING;
}

////////////////////////////////////////////////////////////////

void RepairDesign::makeRepeater(const char* reason,
                                const Point& loc,
                                LibertyCell* buffer_cell,
                                bool resize,
                                int level,
                                // Return values.
                                PinSeq& load_pins,
                                float& repeater_cap,
                                float& repeater_fanout,
                                float& repeater_max_slew)
{ 
  //YL
  if (store_buffered_trees_flag_ == true) {
    float drvr_output_cap=0.0;
    float drvr_output_slew=0.0;
    makeRepeater_update(reason,
                        loc,
                        buffer_cell,
                        resize,
                        level,
                        load_pins,
                        repeater_cap,
                        repeater_fanout,
                        repeater_max_slew,
                        drvr_output_cap,
                        drvr_output_slew);
    return;
  }
  Net* out_net;
  Pin *repeater_in_pin, *repeater_out_pin;
  makeRepeater(reason,
               loc.getX(),
               loc.getY(),
               buffer_cell,
               resize,
               level,
               load_pins,
               repeater_cap,
               repeater_fanout,
               repeater_max_slew,
               out_net,
               repeater_in_pin,
               repeater_out_pin);
}

////////////////////////////////////////////////////////////////
void RepairDesign::makeRepeater(const char* reason,
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
                                Pin*& repeater_out_pin)
{
  if (store_buffered_trees_flag_ == true) {
    float drvr_output_cap=0.0;
    float drvr_output_slew=0.0;
    
    makeRepeater_update(reason,
                        x,
                        y,
                        buffer_cell,
                        resize,
                        level,
                        load_pins,
                        repeater_cap,
                        repeater_fanout,
                        repeater_max_slew,
                        out_net,
                        repeater_in_pin,
                        repeater_out_pin,
                        drvr_output_cap,
                        drvr_output_slew);
    return;
  }
 
  LibertyPort *buffer_input_port, *buffer_output_port;
  buffer_cell->bufferPorts(buffer_input_port, buffer_output_port);
  string buffer_name = resizer_->makeUniqueInstName(reason);
  debugPrint(logger_,
             RSZ,
             "repair_net",
             2,
             "{:{}s}{} {} {} ({} {})",
             "",
             level,
             reason,
             buffer_name.c_str(),
             buffer_cell->name(),
             units_->distanceUnit()->asString(dbuToMeters(x), 1),
             units_->distanceUnit()->asString(dbuToMeters(y), 1));

  // Inserting a buffer is complicated by the fact that verilog netlists
  // use the net name for input and output ports. This means the ports
  // cannot be moved to a different net.

  // This cannot depend on the net in caller because the buffer may be inserted
  // between the driver and the loads changing the net as the repair works its
  // way from the loads to the driver.

  Net *net = nullptr, *in_net;
  bool preserve_outputs = false;
  for (const Pin* pin : load_pins) {
    if (network_->isTopLevelPort(pin)) {
      net = network_->net(network_->term(pin));
      if (network_->direction(pin)->isAnyOutput()) {
        preserve_outputs = true;
        break;
      }
    } else {
      net = network_->net(pin);
      Instance* inst = network_->instance(pin);
      if (resizer_->dontTouch(inst)) {
        preserve_outputs = true;
        break;
      }
    }
  }
  Instance* parent = db_network_->topInstance();

  // If the net is driven by an input port,
  // use the net as the repeater input net so the port stays connected to it.
  if (hasInputPort(net) || !preserve_outputs) {
    in_net = net;
    out_net = resizer_->makeUniqueNet();
    // Copy signal type to new net.
    dbNet* out_net_db = db_network_->staToDb(out_net);
    dbNet* in_net_db = db_network_->staToDb(in_net);
    out_net_db->setSigType(in_net_db->getSigType());

    // Move load pins to out_net.
    for (const Pin* pin : load_pins) {
      Port* port = network_->port(pin);
      Instance* inst = network_->instance(pin);

      // do not disconnect/reconnect don't touch instances
      if (resizer_->dontTouch(inst)) {
        continue;
      }
      sta_->disconnectPin(const_cast<Pin*>(pin));
      sta_->connectPin(inst, port, out_net);
    }
  } else {
    // One of the loads is an output port.
    // Use the net as the repeater output net so the port stays connected to it.
    in_net = resizer_->makeUniqueNet();
    out_net = net;
    // Copy signal type to new net.
    dbNet* out_net_db = db_network_->staToDb(out_net);
    dbNet* in_net_db = db_network_->staToDb(in_net);
    in_net_db->setSigType(out_net_db->getSigType());

    // Move non-repeater load pins to in_net.
    PinSet load_pins1(db_network_);
    for (const Pin* pin : load_pins) {
      load_pins1.insert(pin);
    }

    NetPinIterator* pin_iter = network_->pinIterator(out_net);
    while (pin_iter->hasNext()) {
      const Pin* pin = pin_iter->next();
      if (!load_pins1.hasKey(pin)) {
        Port* port = network_->port(pin);
        Instance* inst = network_->instance(pin);
        sta_->disconnectPin(const_cast<Pin*>(pin));
        sta_->connectPin(inst, port, in_net);
      }
    }
  }

  Point buf_loc(x, y);
  Instance* buffer
      = resizer_->makeBuffer(buffer_cell, buffer_name.c_str(), parent, buf_loc);
  inserted_buffer_count_++;

  sta_->connectPin(buffer, buffer_input_port, in_net);
  sta_->connectPin(buffer, buffer_output_port, out_net);

  resizer_->parasiticsInvalid(in_net);
  resizer_->parasiticsInvalid(out_net);

  // Resize repeater as we back up by levels.
  if (resize) {
    Pin* buffer_out_pin = network_->findPin(buffer, buffer_output_port);
    resizer_->resizeToTargetSlew(buffer_out_pin);
    buffer_cell = network_->libertyCell(buffer);
    buffer_cell->bufferPorts(buffer_input_port, buffer_output_port);
  }

  repeater_in_pin = network_->findPin(buffer, buffer_input_port);
  repeater_out_pin = network_->findPin(buffer, buffer_output_port);

  load_pins.clear();
  load_pins.push_back(repeater_in_pin);
  repeater_cap = resizer_->portCapacitance(buffer_input_port, corner_);
  repeater_fanout = resizer_->portFanoutLoad(buffer_input_port);
  repeater_max_slew = bufferInputMaxSlew(buffer_cell, corner_);
}






bool RepairDesign::hasInputPort(const Net* net)
{
  bool has_top_level_port = false;
  NetConnectedPinIterator* pin_iter = network_->connectedPinIterator(net);
  while (pin_iter->hasNext()) {
    const Pin* pin = pin_iter->next();
    if (network_->isTopLevelPort(pin)
        && network_->direction(pin)->isAnyInput()) {
      has_top_level_port = true;
      break;
    }
  }
  delete pin_iter;
  return has_top_level_port;
}

LibertyCell* RepairDesign::findBufferUnderSlew(float max_slew, float load_cap)
{
  LibertyCell* min_slew_buffer = resizer_->buffer_lowest_drive_;
  float min_slew = INF;
  LibertyCellSeq swappable_cells
      = resizer_->getSwappableCells(resizer_->buffer_lowest_drive_);
  if (!swappable_cells.empty()) {
    sort(swappable_cells,
         [this](const LibertyCell* buffer1, const LibertyCell* buffer2) {
           return resizer_->bufferDriveResistance(buffer1)
                  > resizer_->bufferDriveResistance(buffer2);
         });
    for (LibertyCell* buffer : swappable_cells) {
      if (!resizer_->dontUse(buffer) && resizer_->isLinkCell(buffer)) {
        float slew = resizer_->bufferSlew(
            buffer, load_cap, resizer_->tgt_slew_dcalc_ap_);
        debugPrint(logger_,
                   RSZ,
                   "buffer_under_slew",
                   1,
                   "{:{}s}pt ({} {})",
                   buffer->name(),
                   units_->timeUnit()->asString(slew));
        if (slew < max_slew) {
          return buffer;
        }
        if (slew < min_slew) {
          min_slew_buffer = buffer;
          min_slew = slew;
        }
      }
    }
  }
  // Could not find a buffer under max_slew but this is min slew achievable.
  return min_slew_buffer;
}

double RepairDesign::dbuToMeters(int dist) const
{
  return dist / (dbu_ * 1e+6);
}

int RepairDesign::metersToDbu(double dist) const
{
  return dist * dbu_ * 1e+6;
}

void RepairDesign::printProgress(int iteration,
                                 bool force,
                                 bool end,
                                 int repaired_net_count) const
{
  const bool start = iteration == 0;

  if (start && !end) {
    logger_->report(
        "Iteration | Resized | Buffers | Nets repaired | Remaining");
    logger_->report(
        "---------------------------------------------------------");
  }

  if (iteration % print_interval_ == 0 || force || end) {
    const int nets_left = resizer_->level_drvr_vertices_.size() - iteration;

    std::string itr_field = fmt::format("{}", iteration);
    if (end) {
      itr_field = "final";
    }

    logger_->report("{: >9s} | {: >7d} | {: >7d} | {: >13d} | {: >9d}",
                    itr_field,
                    resize_count_,
                    inserted_buffer_count_,
                    repaired_net_count,
                    nets_left);
  }

  if (end) {
    logger_->report(
        "---------------------------------------------------------");
  }
}



// ---------------------------------------------------------------------------------------
// Update since 2025/01/27
// ---------------------------------------------------------------------------------------


void RepairDesign::initBufferedTrees()
{
  buffered_trees_pre_.clear();
  buffered_trees_post_.clear();
  buffered_trees_mid_.clear();
  buffered_trees_prob_.clear();
}

void RepairDesign::writeBufferedTrees(const std::string& file_name)
{
  std::cout << "This is a dummy function" << std::endl;
}


void RepairDesign::repairDesign_update(
  double max_wire_length,  // zero for none (meters)
  double slew_margin,
  double cap_margin,
  double buffer_gain,
  bool verbose,
  int& repaired_net_count,
  int& slew_violations,
  int& cap_violations,
  int& fanout_violations,
  int& length_violations)
{
  init();
  initBufferedTrees();
  slew_margin_ = slew_margin;
  cap_margin_ = cap_margin;
  buffer_gain_ = buffer_gain;
  bool gain_buffering = (buffer_gain_ != 0.0);

  slew_violations = 0;
  cap_violations = 0;
  fanout_violations = 0;
  length_violations = 0;
  repaired_net_count = 0;
  inserted_buffer_count_ = 0;
  resize_count_ = 0;
  resizer_->resized_multi_output_insts_.clear();

  // keep track of user annotations so we don't remove them
  std::set<std::pair<Vertex*, int>> slew_user_annotated;

  if (gain_buffering) {
    // If gain-based buffering is enabled, we need to override slews in order to
    // get good required time estimates.
    for (int i = resizer_->level_drvr_vertices_.size() - 1; i >= 0; i--) {
      Vertex* drvr = resizer_->level_drvr_vertices_[i];
      for (auto rf : {RiseFall::rise(), RiseFall::fall()}) {
        if (!drvr->slewAnnotated(rf, min_) && !drvr->slewAnnotated(rf, max_)) {
          sta_->setAnnotatedSlew(drvr,
                                 resizer_->tgt_slew_corner_,
                                 sta::MinMaxAll::all(),
                                 rf->asRiseFallBoth(),
                                 resizer_->tgt_slews_[rf->index()]);
        } else {
          slew_user_annotated.insert(std::make_pair(drvr, rf->index()));
        }
      }
    }
    findBufferSizes();
  }

  sta_->checkSlewLimitPreamble();
  sta_->checkCapacitanceLimitPreamble();
  sta_->checkFanoutLimitPreamble();
  sta_->searchPreamble();
  search_->findAllArrivals();

  resizer_->incrementalParasiticsBegin();
  int print_iteration = 0;
  if (resizer_->level_drvr_vertices_.size() > size_t(5) * max_print_interval_) {
    print_interval_ = max_print_interval_;
  } else {
    print_interval_ = min_print_interval_;
  }
  if (verbose) {
    printProgress(print_iteration, false, false, repaired_net_count);
  }
  int max_length = resizer_->metersToDbu(max_wire_length);
  for (int i = resizer_->level_drvr_vertices_.size() - 1; i >= 0; i--) {
    print_iteration++;
    if (verbose) {
      printProgress(print_iteration, false, false, repaired_net_count);
    }
    Vertex* drvr = resizer_->level_drvr_vertices_[i];
    Pin* drvr_pin = drvr->pin();
    Net* net = network_->isTopLevelPort(drvr_pin)
                   ? network_->net(network_->term(drvr_pin))
                   // hier fix
                   : db_network_->dbToSta(db_network_->flatNet(drvr_pin));
    dbNet* net_db = db_network_->staToDb(net);
    bool debug = (drvr_pin == resizer_->debug_pin_);
    if (debug) {
      logger_->setDebugLevel(RSZ, "repair_net", 3);
    }
    if (gain_buffering) {
      search_->findRequireds(drvr->level() + 1);
    }
    if (net && !resizer_->dontTouch(net) && !net_db->isConnectedByAbutment()
        && !sta_->isClock(drvr_pin)
        // Exclude tie hi/low cells and supply nets.
        && !drvr->isConstant()) {
      // To do: create a buffered tree to store the original net
      float max_cap = INF;
      createPreBufferedTree(net, drvr_pin, max_cap);
      // std::cout << "Begin Pre Tree Print\n";
      // buffered_trees_pre_.back()->printTree();
      // std::cout << "End Pre Tree Print\n";
      buffered_trees_post_.push_back(std::make_shared<BufferedTree>(*buffered_trees_pre_.back()));
      buffer_order_count_ = 0; // recored the order of the inserted buffers
      repairNet_update(net,
                drvr_pin,
                drvr,
                true,
                true,
                true,
                max_length,
                true,
                repaired_net_count,
                slew_violations,
                cap_violations,
                fanout_violations,
                length_violations);
    }
    if (debug) {
      logger_->setDebugLevel(RSZ, "repair_net", 0);
    }

    if (gain_buffering) {
      for (auto mm : sta::MinMaxAll::all()->range()) {
        for (auto rf : sta::RiseFallBoth::riseFall()->range()) {
          if (!slew_user_annotated.count(std::make_pair(drvr, rf->index()))) {
            const DcalcAnalysisPt* dcalc_ap
                = resizer_->tgt_slew_corner_->findDcalcAnalysisPt(mm);
            drvr->setSlewAnnotated(false, rf, dcalc_ap->index());
          }
        }
      }
    }
  }

  resizer_->updateParasitics();
  if (verbose) {
    printProgress(print_iteration, true, true, repaired_net_count);
  }
  resizer_->incrementalParasiticsEnd();

  if (inserted_buffer_count_ > 0) {
    resizer_->level_drvr_vertices_valid_ = false;
  }
}


void RepairDesign::repairNet_update(Net* net,
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
                             int& length_violations)
{
  // update the flag
  store_buffered_trees_flag_ = true;
  auto OpenSTASt = std::chrono::high_resolution_clock::now();
  // logger_->report("[MAX Wirelength] : {}", max_length);  //YL 
  // Hands off special nets.
  if (!db_network_->isSpecial(net)) {
    debugPrint(logger_,
               RSZ,
               "repair_net",
               1,
               "repair net {}",
               sdc_network_->pathName(drvr_pin));
    const Corner* corner = sta_->cmdCorner();
    bool repaired_net = false;

    if (buffer_gain_ != 0.0) {
      float fanout, max_fanout, fanout_slack;
      sta_->checkFanout(drvr_pin, max_, fanout, max_fanout, fanout_slack);

      int resized = resizer_->resizeToTargetSlew(drvr_pin);
      if (performGainBuffering(net, drvr_pin, max_fanout)) {
        repaired_net = true;
      }
      // Resize again post buffering as the load changed
      resized += resizer_->resizeToTargetSlew(drvr_pin);
      if (resized > 0) {
        repaired_net = true;
        resize_count_ += 1;
      }
    }

    if (check_fanout) {
      float fanout, max_fanout, fanout_slack;
      sta_->checkFanout(drvr_pin, max_, fanout, max_fanout, fanout_slack);
      
      if (max_fanout > 0.0 && fanout_slack < 0.0) {
        fanout_violations++;
        repaired_net = true;

        debugPrint(logger_, RSZ, "repair_net", 3, "fanout violation");
        LoadRegion region = findLoadRegions(drvr_pin, max_fanout);
        // logger_->report("[MAX Fanout] : {}", max_fanout);  //YL  
        //exit(1);
        
        corner_ = corner;
        makeRegionRepeaters(region,
                            max_fanout,
                            1,
                            drvr_pin,
                            check_slew,
                            check_cap,
                            max_length,
                            resize_drvr);
      }
    }

    // Resize the driver to normalize slews before repairing limit violations.
    if (parasitics_src_ == ParasiticsSrc::placement && resize_drvr) {
      resize_count_ += resizer_->resizeToTargetSlew(drvr_pin);
    }
    // For tristate nets all we can do is resize the driver.
    if (!resizer_->isTristateDriver(drvr_pin)) {
      BufferedNetPtr bnet = resizer_->makeBufferedNetSteiner(drvr_pin, corner);
      if (bnet) {
        resizer_->ensureWireParasitic(drvr_pin, net);
        graph_delay_calc_->findDelays(drvr);

        float max_cap = INF;
        int wire_length = bnet->maxLoadWireLength();
        bool need_repair = needRepair(drvr_pin,
                                      corner,
                                      max_length,
                                      wire_length,
                                      check_cap,
                                      check_slew,
                                      max_cap,
                                      slew_violations,
                                      cap_violations,
                                      length_violations);

        if (need_repair) {
          if (parasitics_src_ == ParasiticsSrc::global_routing && resize_drvr) {
            resize_count_ += resizer_->resizeToTargetSlew(drvr_pin);
            wire_length = bnet->maxLoadWireLength();
            need_repair = needRepair(drvr_pin,
                                     corner,
                                     max_length,
                                     wire_length,
                                     check_cap,
                                     check_slew,
                                     max_cap,
                                     slew_violations,
                                     cap_violations,
                                     length_violations);
          }
          if (need_repair) {
            Point drvr_loc = db_network_->location(drvr->pin());
            debugPrint(
                logger_,
                RSZ,
                "repair_net",
                1,
                "driver {} ({} {}) l={}",
                sdc_network_->pathName(drvr_pin),
                units_->distanceUnit()->asString(dbuToMeters(drvr_loc.getX()),
                                                 1),
                units_->distanceUnit()->asString(dbuToMeters(drvr_loc.getY()),
                                                 1),
                units_->distanceUnit()->asString(dbuToMeters(wire_length), 1));

            // ----------------------------------------------------------------
            // YL: recored problematic nets to files for MLBuf / hacky processing
            auto OpenSTAEnd = std::chrono::high_resolution_clock::now();
            auto OpenSTADuration = std::chrono::duration_cast<std::chrono::milliseconds>(OpenSTAEnd - OpenSTASt);
            
            BufferedTreePtr& problematic_net = buffered_trees_pre_.back();
            auto copied_tree = std::make_shared<rsz::BufferedTree>(*problematic_net); // Make a deep copy
            buffered_trees_prob_.push_back(copied_tree);

            auto tree = copied_tree;
           
            BufferedTreeNodePtr root = tree->getRoot();
            // if (root) {
            //   dbNet* net_db = db_network_->staToDb(tree->getNet());
            //   std::string net_name = net_db->getName();
            //   int node_count = countNodes(root);
            //   std::string runtime_file = std::getenv("TIMERSZ");
            //   std::ofstream file(runtime_file, std::ios::app);
            //   if (file.is_open()) {
            //       file << net_name << "," <<node_count<<","<< OpenSTADuration.count() << ","; // Append the number and a newline
            //       file.close();
            //       std::cout << "netlistModifyDuration appended to file successfully." << std::endl;
            //   } else {
            //       std::cerr << "Unable to open recordORTime.txt" << std::endl;
            //       return; // error code
            //   }
            // }
              

            
            auto repairSt = std::chrono::high_resolution_clock::now();
            

            // obtain net name
            // dbNet* net_db = db_network_->staToDb(problematic_net->getNet());
            // std::string net_name = net_db->getName();
            // logger_->report("[Problematic Net Name] : {}", net_name);  //YL
            // const char* buf_approach = std::getenv("BUF_APPROACH");
            // const std::string default_approach = "rsz";
            // std::string buffer_approach = (buf_approach != nullptr) ? std::string(buf_approach) : default_approach;
            // if (buffer_approach == "MLBuf") {
            //   logger_->report("[INFO] MLBuf is used");
              
            // } else if (buffer_approach == "Hack-y") {
            //     logger_->report("[INFO] Hack-y baseline is used");
            // }
            // else {
            //   logger_->report("[INFO] OR rsz is used");
            // }

            // -------------------------------------------------------
            repairNet(bnet, drvr_pin, max_cap, max_length, corner);
            repaired_net = true;

            if (resize_drvr) {
              resize_count_ += resizer_->resizeToTargetSlew(drvr_pin);
            }
            // auto repairEd = std::chrono::high_resolution_clock::now();
            // auto repairDuration = std::chrono::duration_cast<std::chrono::milliseconds>(repairEd - repairSt);
            // std::string runtime_file2 = std::getenv("TIMERSZ");
            // std::ofstream file2(runtime_file2, std::ios::app);
            // if (file2.is_open()) {
            //     file2 << repairDuration.count() << "\n";  
            //     file2.close();
            //     std::cout << "repairDuration appended to file successfully." << std::endl;
            // } else {
            //     std::cerr << "Unable to open recordORTime.txt (when writing repairDuration)" << std::endl;
            //     return;
            // }

          }
        }
      }
    }
    
    if (repaired_net) {

      // reapir one net --> construct one corresponding post buffered tree
      // buffered_trees_post_.push_back(buffered_tree_);
      // std::cout << "Begin Post Tree Print\n";
      // buffered_trees_post_.back()->printTree();
      // buffered_tree_->printTree();
      // std::cout << "End Post Tree Print\n";
      // std::cout << "Save Post Tree Print\n";
      repaired_net_count++;
      // std::cout << "------------Repaired net count------------- " << repaired_net_count << std::endl;
    }
  }
  store_buffered_trees_flag_ = false;
}


//YL
void RepairDesign::makeRepeater_update(const char* reason,
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
                                float drvr_output_slew)
{ 

  Net* out_net;
  Pin *repeater_in_pin, *repeater_out_pin;
  makeRepeater_update(reason,
               loc.getX(),
               loc.getY(),
               buffer_cell,
               resize,
               level,
               load_pins,
               repeater_cap,
               repeater_fanout,
               repeater_max_slew,
               out_net,
               repeater_in_pin,
               repeater_out_pin,
               drvr_output_cap, //YL
               drvr_output_slew //YL
               );
}

void RepairDesign::makeRepeater_update(const char* reason,
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
                                float drvr_output_cap,
                                float drvr_output_slew)
{
  LibertyPort *buffer_input_port, *buffer_output_port;
  buffer_cell->bufferPorts(buffer_input_port, buffer_output_port);
  string buffer_name = resizer_->makeUniqueInstName(reason);
  debugPrint(logger_,
             RSZ,
             "repair_net",
             2,
             "{:{}s}{} {} {} ({} {})",
             "",
             level,
             reason,
             buffer_name.c_str(),
             buffer_cell->name(),
             units_->distanceUnit()->asString(dbuToMeters(x), 1),
             units_->distanceUnit()->asString(dbuToMeters(y), 1));

  // Inserting a buffer is complicated by the fact that verilog netlists
  // use the net name for input and output ports. This means the ports
  // cannot be moved to a different net.

  // This cannot depend on the net in caller because the buffer may be inserted
  // between the driver and the loads changing the net as the repair works its
  // way from the loads to the driver.

  Net *net = nullptr, *in_net;
  bool preserve_outputs = false;
  for (const Pin* pin : load_pins) {
    if (network_->isTopLevelPort(pin)) {
      net = network_->net(network_->term(pin));
      if (network_->direction(pin)->isAnyOutput()) {
        preserve_outputs = true;
        break;
      }
    } else {
      net = network_->net(pin);
      Instance* inst = network_->instance(pin);
      if (resizer_->dontTouch(inst)) {
        preserve_outputs = true;
        break;
      }
    }
  }
  Instance* parent = db_network_->topInstance();

  // If the net is driven by an input port,
  // use the net as the repeater input net so the port stays connected to it.
  if (hasInputPort(net) || !preserve_outputs) {
    in_net = net;
    out_net = resizer_->makeUniqueNet();
    // Copy signal type to new net.
    dbNet* out_net_db = db_network_->staToDb(out_net);
    dbNet* in_net_db = db_network_->staToDb(in_net);
    out_net_db->setSigType(in_net_db->getSigType());

    // Move load pins to out_net.
    for (const Pin* pin : load_pins) {
      Port* port = network_->port(pin);
      Instance* inst = network_->instance(pin);

      // do not disconnect/reconnect don't touch instances
      if (resizer_->dontTouch(inst)) {
        continue;
      }
      sta_->disconnectPin(const_cast<Pin*>(pin));
      sta_->connectPin(inst, port, out_net);
    }
  } else {
    // One of the loads is an output port.
    // Use the net as the repeater output net so the port stays connected to it.
    in_net = resizer_->makeUniqueNet();
    out_net = net;
    // Copy signal type to new net.
    dbNet* out_net_db = db_network_->staToDb(out_net);
    dbNet* in_net_db = db_network_->staToDb(in_net);
    in_net_db->setSigType(out_net_db->getSigType());

    // Move non-repeater load pins to in_net.
    PinSet load_pins1(db_network_);
    for (const Pin* pin : load_pins) {
      load_pins1.insert(pin);
    }

    NetPinIterator* pin_iter = network_->pinIterator(out_net);
    while (pin_iter->hasNext()) {
      const Pin* pin = pin_iter->next();
      if (!load_pins1.hasKey(pin)) {
        Port* port = network_->port(pin);
        Instance* inst = network_->instance(pin);
        sta_->disconnectPin(const_cast<Pin*>(pin));
        sta_->connectPin(inst, port, in_net);
      }
    }
  }

  Point buf_loc(x, y);
  Instance* buffer
      = resizer_->makeBuffer(buffer_cell, buffer_name.c_str(), parent, buf_loc);
  inserted_buffer_count_++;

  // --------- save buf_loc to file if BUF_APPROACH is rsz ------- //
  dbInst* db_inst = db_network_->staToDb(buffer);
  odb::dbMaster* buf_master = db_inst->getMaster();
  const int buf_width = buf_master->getWidth();
  const int buf_height = buf_master->getHeight();
  int ux = x + buf_width;
  int uy = y + buf_height;
  int buf_area = buf_width * buf_height;

  std::string buf_approach = std::getenv("BUF_APPROACH");
  if (buf_approach == "rsz"){
    const char* output_file = std::getenv("OUTPUT");
    std::ofstream outfile(output_file, std::ios::app);
    if (!outfile.is_open()) {
        std::cerr << "can not open rsz_buf_save.csv" << std::endl;
        exit(1);
    }
    outfile << x << "," << y << "," << ux << "," << uy << "," << buf_area << std::endl;
    outfile.close();
    }
  // ------------------------------------------------------
  
  sta_->connectPin(buffer, buffer_input_port, in_net);
  sta_->connectPin(buffer, buffer_output_port, out_net);


  resizer_->parasiticsInvalid(in_net);
  resizer_->parasiticsInvalid(out_net);

  if (resize) {
    Pin* buffer_out_pin = network_->findPin(buffer, buffer_output_port);
    resizer_->resizeToTargetSlew(buffer_out_pin);
    buffer_cell = network_->libertyCell(buffer);
    buffer_cell->bufferPorts(buffer_input_port, buffer_output_port);
  }

  repeater_in_pin = network_->findPin(buffer, buffer_input_port);
  repeater_out_pin = network_->findPin(buffer, buffer_output_port);
  
  
  repeater_cap = resizer_->portCapacitance(buffer_input_port, corner_);
  repeater_fanout = resizer_->portFanoutLoad(buffer_input_port);
  repeater_max_slew = bufferInputMaxSlew(buffer_cell, corner_);
  
  // logger_->report("--------Add a buffer and the buffer count is ----" + std::to_string(inserted_buffer_count_));
  // Save original features
  if (buffer_order_count_ == 0){
    BufferedTreePtr& tree = buffered_trees_post_.back(); //YL: get the last tree
    // Make a deep copy
    auto copied_tree = std::make_shared<rsz::BufferedTree>(*tree);
    // Push the copy into buffered_trees_mid_
    buffered_trees_mid_.push_back(copied_tree);

  }
  buffer_order_count_++; //YL: track the feature updating when insert a new buffer
  // logger_->report("Begin createPostBufferedTree...\n");
  createPostBufferedTree(
    reason,
    buffer, 
    buffer_cell, 
    x, y, 
    load_pins,
    repeater_fanout,
    repeater_in_pin,
    repeater_out_pin,
    repeater_cap,
    drvr_output_cap,
    drvr_output_slew);
  // logger_->report("End createPostBufferedTree...\n");
  // // for testing
  // std::cout<<"Start testing buffered tree...\n";
  // testBufferedTree(repeater_in_pin,repeater_out_pin, in_net, out_net);
  // std::cout<<"End testing buffered tree...\n";
  
  load_pins.clear();
  load_pins.push_back(repeater_in_pin);
}




void RepairDesign::repairNetWire_update(
    const BufferedNetPtr& bnet,
    int level,
    // Return values.
    // Remaining parasiics after repeater insertion.
    int& wire_length,  // dbu
    PinSeq& load_pins)
{
  debugPrint(logger_,
             RSZ,
             "repair_net",
             3,
             "{:{}s}{}",
             "",
             level,
             bnet->to_string(resizer_));
  int wire_length_ref;
  repairNet(bnet->ref(), level + 1, wire_length_ref, load_pins);
  float max_load_slew = bnet->maxLoadSlew();
  float max_load_slew_margined
      = max_load_slew;  // maxSlewMargined(max_load_slew);

  Point to_loc = bnet->ref()->location();
  int to_x = to_loc.getX();
  int to_y = to_loc.getY();
  Point from_loc = bnet->location();
  int length = Point::manhattanDistance(from_loc, to_loc);
  wire_length = wire_length_ref + length;
  int from_x = from_loc.getX();
  int from_y = from_loc.getY();
  debugPrint(logger_,
             RSZ,
             "repair_net",
             3,
             "{:{}s}wl={} l={}",
             "",
             level,
             units_->distanceUnit()->asString(dbuToMeters(wire_length), 1),
             units_->distanceUnit()->asString(dbuToMeters(length), 1));
  double length1 = dbuToMeters(length);
  double wire_res, wire_cap;
  bnet->wireRC(corner_, resizer_, wire_res, wire_cap);
  // std::cout<<"[lyt test in repairNetWire_update] wire_res: "<<wire_res<<"wire_cap: "<<wire_cap<<std::endl;
  // ref_cap includes ref's wire cap
  double ref_cap = bnet->ref()->cap();
  double load_cap = length1 * wire_cap + ref_cap;

  float r_drvr = resizer_->driveResistance(drvr_pin_);
  double load_slew = (r_drvr + dbuToMeters(wire_length) * wire_res) * load_cap
                     * elmore_skew_factor_;
  debugPrint(logger_,
             RSZ,
             "repair_net",
             3,
             "{:{}s}load_slew={} r_drvr={}",
             "",
             level,
             delayAsString(load_slew, this, 3),
             units_->resistanceUnit()->asString(r_drvr, 3));

  LibertyCell* buffer_cell = resizer_->findTargetCell(
      resizer_->buffer_lowest_drive_, load_cap, false);
  bnet->setCapacitance(load_cap);
  bnet->setFanout(bnet->ref()->fanout());

  // Check that the slew limit specified is within the bounds of reason.
  pre_checks_->checkSlewLimit(ref_cap, max_load_slew);
  //============================================================================
  // Back up from pt to from_pt adding repeaters as necessary for
  // length/max_cap/max_slew violations.
  while ((max_length_ > 0 && wire_length > max_length_)
         || (wire_cap > 0.0 && max_cap_ > 0.0 && load_cap > max_cap_)
         || load_slew > max_load_slew_margined) {
    // Make the wire a bit shorter than necessary to allow for
    // offset from instance origin to pin and detailed placement movement.
    constexpr double length_margin = .05;
    bool split_wire = false;
    bool resize = true;
    // Distance from repeater to ref_.
    //              length
    // from----------------------------to/ref
    //
    //                     split_length
    // from-------repeater-------------to/ref
    int split_length = std::numeric_limits<int>::max();
    if (max_length_ > 0 && wire_length > max_length_) {
      debugPrint(logger_,
                 RSZ,
                 "repair_net",
                 3,
                 "{:{}s}max wire length violation {} > {}",
                 "",
                 level,
                 units_->distanceUnit()->asString(dbuToMeters(wire_length), 1),
                 units_->distanceUnit()->asString(dbuToMeters(max_length_), 1));
      split_length = min(max(max_length_ - wire_length_ref, 0), length / 2);

      split_wire = true;
    }
    if (wire_cap > 0.0 && load_cap > max_cap_) {
      debugPrint(logger_,
                 RSZ,
                 "repair_net",
                 3,
                 "{:{}s}max cap violation {} > {}",
                 "",
                 level,
                 units_->capacitanceUnit()->asString(load_cap, 3),
                 units_->capacitanceUnit()->asString(max_cap_, 3));
      if (ref_cap > max_cap_) {
        split_length = 0;
        split_wire = false;
      } else {
        split_length
            = min(split_length, metersToDbu((max_cap_ - ref_cap) / wire_cap));
        split_wire = true;
      }
    }
    if (load_slew > max_load_slew_margined) {
      debugPrint(logger_,
                 RSZ,
                 "repair_net",
                 3,
                 "{:{}s}max load slew violation {} > {}",
                 "",
                 level,
                 delayAsString(load_slew, this, 3),
                 delayAsString(max_load_slew_margined, this, 3));
      // Using elmore delay to approximate wire
      // load_slew = (Rbuffer + (L+Lref)*Rwire) * (L*Cwire + Cref) *
      // elmore_skew_factor_ Setting this to max_load_slew_margined is a
      // quadratic in L L^2*Rwire*Cwire + L*(Rbuffer*Cwire + Rwire*Cref +
      // Rwire*Cwire*Lref)
      //   + Rbuffer*Cref - max_load_slew_margined/elmore_skew_factor_
      // Solve using quadradic eqn for L.
      float wire_length_ref1 = dbuToMeters(wire_length_ref);
      float r_buffer = resizer_->bufferDriveResistance(buffer_cell);
      float a = wire_res * wire_cap;
      float b = r_buffer * wire_cap + wire_res * ref_cap
                + wire_res * wire_cap * wire_length_ref1;
      float c = r_buffer * ref_cap + wire_length_ref1 * wire_res * ref_cap
                - max_load_slew_margined / elmore_skew_factor_;
      float l = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
      if (l >= 0.0) {
        if (split_length > 0.0) {
          split_length = min(split_length, metersToDbu(l));
        } else {
          split_length = metersToDbu(l);
        }
        split_wire = true;
        resize = false;
      } else {
        split_length = 0;
        split_wire = false;
        resize = true;
      }
    }

    if (split_wire) {
      debugPrint(
          logger_,
          RSZ,
          "repair_net",
          3,
          "{:{}s}split length={}",
          "",
          level,
          units_->distanceUnit()->asString(dbuToMeters(split_length), 1));
      // Distance from to_pt to repeater backward toward from_pt.
      // Note that split_length can be longer than the wire length
      // because it is the maximum value that satisfies max slew/cap.
      double buf_dist = (split_length >= length)
                            ? length
                            : split_length * (1.0 - length_margin);
      double dx = from_x - to_x;
      double dy = from_y - to_y;
      double d = (length == 0) ? 0.0 : buf_dist / length;
      int buf_x = to_x + d * dx;
      int buf_y = to_y + d * dy;
      float repeater_cap, repeater_fanout;

      // makeRepeater_update("wire",
      //              Point(buf_x, buf_y),
      //              buffer_cell,
      //              resize,
      //              level,
      //              load_pins,
      //              repeater_cap,
      //              repeater_fanout,
      //              max_load_slew);

      // --------- YL: update repeater_cap -------------
      LibertyPort *buffer_input_port, *buffer_output_port;
      buffer_cell->bufferPorts(buffer_input_port, buffer_output_port);
      repeater_cap = resizer_->portCapacitance(buffer_input_port, corner_);
      // -------------------------------------------------

      // Update for the next round.
      length -= buf_dist;
      wire_length = length;
      to_x = buf_x;
      to_y = buf_y;

      length1 = dbuToMeters(length);
      wire_length_ref = 0.0;
      load_cap = repeater_cap + length1 * wire_cap; // YL: driver output cap
      ref_cap = repeater_cap;
      load_slew
          = (r_drvr + length1 * wire_res) * load_cap * elmore_skew_factor_;
          //YL: driver output slew
      
      makeRepeater_update("wire",
              Point(buf_x, buf_y),
              buffer_cell,
              resize,
              level,
              load_pins,
              repeater_cap,
              repeater_fanout,
              max_load_slew,
              load_cap,   // driver updated output cap 
              load_slew); // driver updated output slew

      max_load_slew_margined = max_load_slew;  // maxSlewMargined(max_load_slew);
      buffer_cell = resizer_->findTargetCell(
          resizer_->buffer_lowest_drive_, load_cap, false);

      bnet->setCapacitance(load_cap);
      bnet->setFanout(repeater_fanout);
      bnet->setMaxLoadSlew(max_load_slew);

      debugPrint(logger_,
                 RSZ,
                 "repair_net",
                 3,
                 "{:{}s}l={}",
                 "",
                 level,
                 units_->distanceUnit()->asString(length1, 1));
    } else {
      break;
    }
  }
}


void RepairDesign::repairNetJunc_update(
    const BufferedNetPtr& bnet,
    int level,
    // Return values.
    // Remaining parasiics after repeater insertion.
    int& wire_length,  // dbu
    PinSeq& load_pins)
{
  debugPrint(logger_,
             RSZ,
             "repair_net",
             3,
             "{:{}s}{}",
             "",
             level,
             bnet->to_string(resizer_));
  Point loc = bnet->location();
  double wire_res, wire_cap;
  resizer_->wireSignalRC(corner_, wire_res, wire_cap);
  // std::cout<<"[lyt test in repairNetJunc_update] wire_res: "<<wire_res<<"wire_cap: "<<wire_cap<<std::endl;

  BufferedNetPtr left = bnet->ref();
  int wire_length_left = 0;
  PinSeq loads_left;
  repairNet(left, level + 1, wire_length_left, loads_left);
  float cap_left = left->cap();
  float fanout_left = left->fanout();
  float max_load_slew_left = left->maxLoadSlew();

  BufferedNetPtr right = bnet->ref2();
  int wire_length_right = 0;
  PinSeq loads_right;
  repairNet(right, level + 1, wire_length_right, loads_right);
  float cap_right = right->cap();
  float fanout_right = right->fanout();
  float max_load_slew_right = right->maxLoadSlew();

  debugPrint(logger_,
             RSZ,
             "repair_net",
             3,
             "{:{}s}left  l={} cap={} fanout={}",
             "",
             level,
             units_->distanceUnit()->asString(dbuToMeters(wire_length_left), 1),
             units_->capacitanceUnit()->asString(cap_left, 3),
             fanout_left);
  debugPrint(
      logger_,
      RSZ,
      "repair_net",
      3,
      "{:{}s}right l={} cap={} fanout={}",
      "",
      level,
      units_->distanceUnit()->asString(dbuToMeters(wire_length_right), 1),
      units_->capacitanceUnit()->asString(cap_right, 3),
      fanout_right);

  wire_length = wire_length_left + wire_length_right;
  float load_cap = cap_left + cap_right;
  float max_load_slew = min(max_load_slew_left, max_load_slew_right);
  float max_load_slew_margined
      = max_load_slew;  // maxSlewMargined(max_load_slew);
  LibertyCell* buffer_cell = resizer_->findTargetCell(
      resizer_->buffer_lowest_drive_, load_cap, false);

  // Check for violations when the left/right branches are combined.
  // Add a buffer to left or right branch to stay under the max
  // cap/length/fanout.
  bool repeater_left = false;
  bool repeater_right = false;
  float r_drvr = resizer_->driveResistance(drvr_pin_);
  float load_slew = (r_drvr + dbuToMeters(wire_length) * wire_res) * load_cap
                    * elmore_skew_factor_;
  bool load_slew_violation = load_slew > max_load_slew_margined;
  const char* repeater_reason = nullptr;
  // Driver slew checks were converted to max cap.
  if (load_slew_violation) {
    debugPrint(logger_,
               RSZ,
               "repair_net",
               3,
               "{:{}s}load slew violation {} > {}",
               "",
               level,
               delayAsString(load_slew, this, 3),
               delayAsString(max_load_slew_margined, this, 3));
    float slew_left = (r_drvr + dbuToMeters(wire_length_left) * wire_res)
                      * cap_left * elmore_skew_factor_;
    float slew_slack_left = maxSlewMargined(max_load_slew_left) - slew_left;
    float slew_right = (r_drvr + dbuToMeters(wire_length_right) * wire_res)
                       * cap_right * elmore_skew_factor_;
    float slew_slack_right = maxSlewMargined(max_load_slew_right) - slew_right;
    debugPrint(logger_,
               RSZ,
               "repair_net",
               3,
               "{:{}s} slew slack left={} right={}",
               "",
               level,
               delayAsString(slew_slack_left, this, 3),
               delayAsString(slew_slack_right, this, 3));
    // Isolate the branch with the smaller slack.
    if (slew_slack_left < slew_slack_right) {
      repeater_left = true;
    } else {
      repeater_right = true;
    }
    repeater_reason = "load_slew";
  }
  bool cap_violation = (cap_left + cap_right) > max_cap_;
  if (cap_violation) {
    debugPrint(logger_, RSZ, "repair_net", 3, "{:{}s}cap violation", "", level);
    if (cap_left > cap_right) {
      repeater_left = true;
    } else {
      repeater_right = true;
    }
    repeater_reason = "max_cap";
  }
  bool length_violation
      = max_length_ > 0 && (wire_length_left + wire_length_right) > max_length_;
  if (length_violation) {
    debugPrint(
        logger_, RSZ, "repair_net", 3, "{:{}s}length violation", "", level);
    if (wire_length_left > wire_length_right) {
      repeater_left = true;
    } else {
      repeater_right = true;
    }
    repeater_reason = "max_length";
  }
  
  // YL: calculate the driver output cap and slew
  float drvr_output_cap = cap_left + cap_right;
  if(repeater_left) {
    wire_length_left = 0;
  }
  if(repeater_right) {
    wire_length_right = 0;
  }
  float drvr_output_slew = (r_drvr + dbuToMeters(wire_length_left + wire_length_right) * wire_res) * drvr_output_cap
                  * elmore_skew_factor_;

  if (repeater_left) {
    makeRepeater_update(repeater_reason,
                 loc,
                 buffer_cell,
                 true,
                 level,
                 loads_left,
                 cap_left,
                 fanout_left,
                 max_load_slew_left,
                 drvr_output_cap, 
                 drvr_output_slew);
    wire_length_left = 0;
  }
  if (repeater_right) {
    makeRepeater_update(repeater_reason,
                 loc,
                 buffer_cell,
                 true,
                 level,
                 loads_right,
                 cap_right,
                 fanout_right,
                 max_load_slew_right,
                 drvr_output_cap, 
                 drvr_output_slew);
    wire_length_right = 0;
  }

  // Update after left/right repeaters are inserted.
  wire_length = wire_length_left + wire_length_right;

  bnet->setCapacitance(cap_left + cap_right); //driver output cap
  bnet->setFanout(fanout_right + fanout_left);
  bnet->setMaxLoadSlew(min(max_load_slew_left, max_load_slew_right));

  // Union left/right load pins.
  for (const Pin* load_pin : loads_left) {
    load_pins.push_back(load_pin);
  }
  for (const Pin* load_pin : loads_right) {
    load_pins.push_back(load_pin);
  }
}

//YL: verfity the constructed buffered tree
void RepairDesign::testBufferedTree(Pin*& repeater_in_pin,
                        Pin*& repeater_out_pin, 
                        Net*& in_net, 
                        Net*& out_net)
{
  std::string buffer_in_pin_name = sdc_network_->pathName(repeater_in_pin);
  std::string buffer_out_pin_name = sdc_network_->pathName(repeater_out_pin);
  std::cout<<"Buffer in pin name: "<<buffer_in_pin_name<<std::endl;
  std::cout<<"Buffer out pin name: "<<buffer_out_pin_name<<std::endl;
  NetConnectedPinIterator* pin_iter_in = network_->connectedPinIterator(in_net);
  std::cout<<"In pin name: ";
  while (pin_iter_in->hasNext()) {
    const Pin* pin = pin_iter_in->next();
    std::string in_pin_name = sdc_network_->pathName(pin);
    Point sink_loc = db_network_->location(pin);
    float in_pin_x = dbuToMeters(sink_loc.getX()) * 1e6;
    float in_pin_y = dbuToMeters(sink_loc.getY()) * 1e6;
    std::cout<<" "<<in_pin_name<<" ("<<in_pin_x<<", "<<in_pin_y<<") \t";
  }

  NetConnectedPinIterator* pin_iter_out = network_->connectedPinIterator(out_net);
  std::cout<<"\n Out pin name: ";
  while (pin_iter_out->hasNext()) {
    const Pin* pin = pin_iter_out->next();
    std::string out_pin_name = sdc_network_->pathName(pin);
    Point sink_loc = db_network_->location(pin);
    float out_pin_x = dbuToMeters(sink_loc.getX()) * 1e6;
    float out_pin_y = dbuToMeters(sink_loc.getY()) * 1e6;
    std::cout<<" "<<out_pin_name<<" ("<<out_pin_x<<",  "<<out_pin_y<<") \t";
  }

}

                     
// ----------------------------------------------------------------------------
//  New functions for MLBuf
// ----------------------------------------------------------------------------


float RepairDesign::getOutputCap(const Pin* drvr_pin, float& max_cap)
{
  float cap1, max_cap1, cap_slack1;
  const Corner* corner1;
  const RiseFall* tr1;
  sta_->checkCapacitance(drvr_pin, nullptr, max_, corner1, tr1, cap1, max_cap1, cap_slack1);
  if (max_cap1 > 0.0 && corner1) {
    max_cap1 *= (1.0 - cap_margin_ / 100.0);
    max_cap = max_cap1;
    return cap1;
  } 

  return 0.0;
}

void RepairDesign::getSlew(
  const Pin* drvr_pin, 
  float& max_cap,
  float &output_slew,
  float &input_slew)
{
  float slew1, slew_slack1, max_slew1;
  const Corner* corner1;
  // Check slew at the driver.
  checkSlew(drvr_pin, slew1, max_slew1, slew_slack1, corner1);
  // Max slew violations at the driver pin are repaired by reducing the
  // load capacitance. Wire resistance may shield capacitance from the
  // driver but so this is conservative.
  // Find max load cap that corresponds to max_slew.
  LibertyPort* drvr_port = network_->libertyPort(drvr_pin);
  output_slew = slew1;
  if (corner1 && max_slew1 > 0.0) {
    if (drvr_port) {
      float max_cap1 = findSlewLoadCap(drvr_port, max_slew1, corner1);
      max_cap = min(max_cap, max_cap1);
    }
  }
  // Check slew at the loads.
  // Note that many liberty libraries do not have max_transition attributes on
  // input pins.
  // Max slew violations at the load pins are repaired by inserting buffers
  // and reducing the wire length to the load.
  resizer_->checkLoadSlews(
      drvr_pin, slew_margin_, slew1, max_slew1, slew_slack1, corner1);
  input_slew = slew1;
}


// Signoff at 2025/01/29
void RepairDesign::createPreBufferedTree(
  Net* net, 
  const Pin* drvr_pin, 
  float max_cap)
{
  BufferedTreePtr tree = std::make_shared<BufferedTree>(net);
  BufferedTreeNodePtr root = std::make_shared<BufferedTreeNode>();
  root->pin_name = sdc_network_->pathName(drvr_pin);
  Point drvr_loc = db_network_->location(drvr_pin);
  root->x = dbuToMeters(drvr_loc.getX()) * 1e6;
  root->y = dbuToMeters(drvr_loc.getY()) * 1e6;
  root->resistance = resizer_->driveResistance(drvr_pin);
  root->buffer_order = 0;
  root->buffer_type = "-1";
  root->buf_reason = "-1";
  // Check Capacitance
  root->output_cap = getOutputCap(drvr_pin, max_cap);
  root->input_cap = -1.0;

  // check slew
  float input_slew; // for sinks
  float output_slew;
  getSlew(drvr_pin, max_cap, output_slew, input_slew);  
  root->output_slew = output_slew;
  root->input_slew = -1.0;
  root->max_output_cap = max_cap;
  root->parent = nullptr;
  root->node_type = BufferedTreeNodeType::DRIVER;

 
  float slew1, slew_slack1, max_slew1;
  const Corner* corner1;
  // Check slew at the driver.
  checkSlew(drvr_pin, slew1, max_slew1, slew_slack1, corner1);

  if (corner1){ // if corner1 is nullptr (drvr does not related to any timing-related path), we can skip this net
    // Check all the sink pins
    NetConnectedPinIterator* pin_iter = network_->connectedPinIterator(net);
    while (pin_iter->hasNext()) {
      const Pin* pin = pin_iter->next();
      // ignore all the sinks that are "output-ports"
      if (pin != drvr_pin && !network_->isTopLevelPort(pin)
        && network_->direction(pin) == sta::PortDirection::input()
        && network_->libertyPort(pin)) {
        BufferedTreeNodePtr sink = std::make_shared<BufferedTreeNode>();
        sink->pin_name = sdc_network_->pathName(pin);
        Point sink_loc = db_network_->location(pin);
        sink->x = dbuToMeters(sink_loc.getX()) * 1e6;
        sink->y = dbuToMeters(sink_loc.getY()) * 1e6;
        // Check Capacitance
        sink->output_cap = -1.0;
        LibertyPort* input_port = network_->libertyPort(pin);
        sink->input_cap = resizer_->portCapacitance(input_port, corner1);
        // check slew
        sink->output_slew = -1.0;
        sink->input_slew = input_slew;
        sink->max_output_cap = -1.0;
        sink->parent = root;
        sink->node_type = BufferedTreeNodeType::SINK;
        sink->resistance = -1;
        sink->buffer_type = "-1";
        sink->buf_reason = "-1";
        sink->buffer_order = 0;
        root->children.push_back(sink);
      }
    } 
  }
  else {
    logger_->report("[Warning] Failed to retrieve corner for driver pin: {}", sdc_network_->pathName(drvr_pin));    
  }
  root->fanout = root->children.size();
  tree->setRoot(root);
  buffered_trees_pre_.push_back(tree);
}


void RepairDesign::createPostBufferedTree(
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
  float drvr_output_slew)
{
  BufferedTreePtr& tree = buffered_trees_post_.back(); //YL: get the last tree
  
  // 1. create buffer node
  BufferedTreeNodePtr buffer = std::make_shared<BufferedTreeNode>();
  buffer->pin_name = sdc_network_->pathName(repeater_in_pin); // buffer cell input pin
  buffer->x = dbuToMeters(x) * 1e6;
  buffer->y = dbuToMeters(y) * 1e6;
  buffer->buffer_order = buffer_order_count_;
  buffer->buffer_type = buffer_cell->name();
  buffer->buf_reason = reason;
  // buffer resistance
  float r_buffer = resizer_->bufferDriveResistance(buffer_cell);
  buffer->resistance = r_buffer;
  // const Corner* corner1;

  // Check Capacitance
  float max_cap = INF;
  //YL: does the buffer need output_cap?
  buffer->output_cap = getOutputCap(repeater_out_pin, max_cap);
  buffer->input_cap = repeater_input_cap;
  
  //YL: does the buffer need output_slew?
  buffer->output_slew = -1.0;
  buffer->input_slew = drvr_output_slew;

  buffer->max_output_cap = max_cap;
  buffer->node_type = BufferedTreeNodeType::BUFFER;

  // 2. add buffer to the child of the root
  BufferedTreeNodePtr root = tree->getRoot();
  buffer->parent = root;
  root->children.push_back(buffer);
  root->buffer_order = buffer_order_count_;
  // YL: Update output cap and output slew of the driver
  if (drvr_output_cap !=0 and drvr_output_slew !=0)
  {
    root->output_cap = drvr_output_cap;
    root->output_slew = drvr_output_slew;
  }
  
  // 3. find sinks driven by the buffer
  std::vector<BufferedTreeNodePtr> nodes_to_remove;
  auto& root_children = tree->getRoot()->children;
  // Precompute load pin names for faster lookup
  std::unordered_set<std::string> load_pin_names;
  for (const Pin* pin : load_pins) {
      load_pin_names.insert(sdc_network_->pathName(pin));
  }
  // Iterate through root's children to find matching loads
  for (auto& child : root_children) {
    child->buffer_order = buffer_order_count_;
      if (load_pin_names.find(child->pin_name) != load_pin_names.end()) {
          buffer->children.push_back(child);
          child->parent = buffer;
          nodes_to_remove.push_back(child);
      }
      //YL: update input slew of the sink
      else if (drvr_output_slew!=0) {
        child->input_slew = drvr_output_slew;
      }
  }

  // 4. Remove loads from the root
  for (const auto& node : nodes_to_remove) {
      root_children.erase(
          std::remove(root_children.begin(), root_children.end(), node),
          root_children.end()
      );
  }

  //5. update fanout
  root->fanout = root->children.size();
  buffer->fanout = buffer->children.size();
  // tree->setRoot(root);
  
  // Make a deep copy
  auto copied_tree = std::make_shared<rsz::BufferedTree>(*tree);

  // Push the copy into buffered_trees_mid_
  buffered_trees_mid_.push_back(copied_tree);

}
                        
// ----------------------------------------------------------------------------
//  Save Buffered Trees
// ----------------------------------------------------------------------------

// count the number of sinks and buffers in the tree
void RepairDesign::countNodes(const BufferedTreePtr& tree, int& sink_count, int& buffer_count) {
  if (!tree || !tree->getRoot()) return;

  std::queue<BufferedTreeNodePtr> q;
  q.push(tree->getRoot());

  sink_count = 0;
  buffer_count = 0;

  while (!q.empty()) {
      BufferedTreeNodePtr node = q.front();
      q.pop();

      if (node->node_type == BufferedTreeNodeType::SINK) {
          sink_count++;
      } else if (node->node_type == BufferedTreeNodeType::BUFFER) {
          buffer_count++;
      }

      for (const auto& child : node->children) {
          q.push(child);
      }
  }
}

// save to csv file
void RepairDesign::printTreeToCSV(const std::vector<BufferedTreePtr>& trees, const std::string& filename, bool buffered_tree_print_flag) {
  std::ofstream file(filename);

  if (!file.is_open()) {
      std::cerr << "Error: Unable to open file for writing!\n";
      return;
  }

  // csv header
  file << "Net Name,Tree ID,Depth,Node ID,Node Type,Pin Name,X,Y,Input Slew,Output Slew,";
  file << "Input Cap,Output Cap,Max Output Cap,Fanout,Resistance,Sink Count,Buffer Count,Parent Node ID,#Children,Buffer Order,Buffer Type, Buffer Reason\n";

  int tree_id = 0;
  for (const auto& tree : trees) {
    if (!tree || !tree->getRoot()) continue;

    // Skip this tree if it does not contain any "Buffer" node
    // if (!treeContainsBuffer(tree) && buffered_tree_print_flag) continue;
    
    // Identify whether the tree contains buf and sinks
    bool hasBuf, hasSink;
    std::tie(hasBuf, hasSink) = treeContainsSpecificNode(tree);
    // Skip this tree if it does not contain any "Buffer" node
    if (!hasBuf && buffered_tree_print_flag) continue;

    // Skip this tree if it does not contain any "Sink" node
    if (!hasSink) continue;
  
    int sink_count = 0, buffer_count = 0;
    countNodes(tree, sink_count, buffer_count);

    // obtain net name
    dbNet* net_db = db_network_->staToDb(tree->getNet());
    std::string net_name = net_db->getName();

    std::queue<std::pair<BufferedTreeNodePtr, int>> q;
    
    q.push({tree->getRoot(), 0});
    int node_id = 0;

    while (!q.empty()) {
      auto [node, depth] = q.front();
      q.pop();
      node->node_id = node_id++;

      file << net_name << ",";                         // Net Name
      file << tree_id << ",";                          // Tree ID
      file << depth << ",";                            // Depth
      file << node->node_id << ",";                    // Node ID
      file << bufferedTreeNodeTypeToString(node->node_type) << ",";  // Node Type
      file << node->pin_name << ",";                   // Pin Name
      file << node->x << "," << node->y << ",";        // X, Y
      file << node->input_slew << "," << node->output_slew << ",";  // Slew
      file << node->input_cap << "," << node->output_cap << ",";    // Capacitance
      file << node->max_output_cap << ",";             // Max Output Cap
      file << node->fanout << ",";                     // Fanout
      file << node->resistance << ",";                     // Resistance
      file << sink_count << "," << buffer_count << ","; // Sink Count, Buffer Count
      file << (node->parent ? std::to_string(node->parent->node_id) : "None") << ",";  // Parent Node ID
      file << node->children.size() << ",";            // #Children
      file << node->buffer_order << "," << node->buffer_type << "," << node->buf_reason <<"\n"; // Buffer Order, Buffer Type, Reason
              
      for (const auto& child : node->children) {
          q.push({child, depth + 1});
      }
    }
    tree_id++;
  }
  file.close();
}

// check if the tree contains buffer or sink 
std::pair<bool, bool> RepairDesign::treeContainsSpecificNode(const BufferedTreePtr& tree) {
  bool buffer_flag = false;
  bool sink_flag = false;
  // driver_flag = false;
  if (!tree || !tree->getRoot()) return {buffer_flag, sink_flag};

  std::queue<BufferedTreeNodePtr> q;
  q.push(tree->getRoot());

  while (!q.empty()) {
      auto node = q.front();
      q.pop();

      if (node->node_type == BufferedTreeNodeType::BUFFER) {
          buffer_flag = true; // Found at least one buffer node
          // return true; 
      }
      if (node->node_type == BufferedTreeNodeType::SINK) {
          sink_flag = true;
          // return true; // Found at least one buffer node
      }
      if (buffer_flag && sink_flag) return {buffer_flag, sink_flag};

      for (const auto& child : node->children) {
          q.push(child);
      }
      
  }
  return {buffer_flag, sink_flag}; // No buffer node found
}

// check if the tree contains buffer
bool RepairDesign::treeContainsBuffer(const BufferedTreePtr& tree) {
    if (!tree || !tree->getRoot()) return false;

    std::queue<BufferedTreeNodePtr> q;
    q.push(tree->getRoot());

    while (!q.empty()) {
        auto node = q.front();
        q.pop();

        if (node->node_type == BufferedTreeNodeType::BUFFER) {
            return true; // Found at least one buffer node
        }

        for (const auto& child : node->children) {
            q.push(child);
        }
    }
    return false; // No buffer node found
}

int RepairDesign::countNodes(const BufferedTreeNodePtr& node) 
{
  if (!node) {
    return 0;
  }
  int count = 1; 
  for (auto& child : node->children) {
    count += countNodes(child);
  }
  return count;
}

// YL: for MLBuf Integration
void RepairDesign::saveProbNets() {
  if (buffered_trees_prob_.size() > 0) {
    bool buffered_tree_print_flag = false;
    const char* buf_approach = std::getenv("BUF_APPROACH");
    const std::string default_approach = "rsz";
    std::string buffer_approach = (buf_approach != nullptr) ? std::string(buf_approach) : default_approach;

    // for (auto& tree : buffered_trees_prob_) {
    //   BufferedTreeNodePtr root = tree->getRoot();
    //   if (root) {
    //     dbNet* net_db = db_network_->staToDb(tree->getNet());
    //     std::string net_name = net_db->getName();
    //     int node_count = countNodes(root);
    //     std::cout << "Net name: " << rnet_name << " has " 
    //               << node_count << " nodes." << std::endl;
    //   }
    // }
    if (buffer_approach == "MLBuf" || buffer_approach == "Hack-y") {
      std::string prob_net_file = std::getenv("INPUT"); // save problematic nets
      printTreeToCSV(buffered_trees_prob_, prob_net_file, buffered_tree_print_flag);
      std::cout << "buffered_trees_prob_ saved to " << prob_net_file << std::endl;
    } 
    
  }
}

// save
void RepairDesign::saveBufferedTrees() {
  // std::cout << "Total Trees (Pre Repair): " << buffered_trees_pre_.size() << std::endl;
  // std::cout << "Total Trees (Post Repair) " << buffered_trees_post_.size() << std::endl;
  // std::cout << "Total Trees (Mid Repair) " << buffered_trees_mid_.size() << std::endl;

  std::string pre_filename = generateUniqueFilename("buffered_trees_pre");
  std::string post_filename = generateUniqueFilename("buffered_trees_post");
  std::string mid_filename = generateUniqueFilename("buffered_trees_mid");

  if (buffered_trees_pre_.size() > 0) {
    bool buffered_tree_print_flag = false;
    printTreeToCSV(buffered_trees_pre_, pre_filename, buffered_tree_print_flag);
    std::cout << "buffered_trees_pre saved to " << pre_filename << std::endl;
  }

  if (buffered_trees_post_.size() > 0) {
    bool buffered_tree_print_flag = true;
    printTreeToCSV(buffered_trees_post_, post_filename, buffered_tree_print_flag);
    std::cout << "buffered_trees_post saved to " << post_filename << std::endl;
  }

  if (buffered_trees_mid_.size() > 0) {
    bool buffered_tree_print_flag = false;
    printTreeToCSV(buffered_trees_mid_, mid_filename, buffered_tree_print_flag);
    std::cout << "buffered_trees_mid saved to " << mid_filename << std::endl;
  }

  // printTreeToCSV(buffered_trees_pre_, "buffered_trees_pre.csv");
  // printTreeToCSV(buffered_trees_post_, "buffered_trees_post.csv");
}

// obtain the current time stamp
std::string RepairDesign::getCurrentTimestamp() {
    std::time_t now = std::time(nullptr);
    std::tm* local_time = std::localtime(&now);
    
    std::ostringstream oss;
    oss << (1900 + local_time->tm_year) 
        << (local_time->tm_mon + 1) 
        << local_time->tm_mday << "_" 
        << local_time->tm_hour 
        << local_time->tm_min 
        << local_time->tm_sec;
    
    return oss.str();
}

std::string RepairDesign::generateUniqueFilename(const std::string& base_name) {
  const std::string default_directory = "/home/dgx_projects/MLBuf/virtual_buffer/DATA/buffered_trees_data/";  
  
  // User-defined directory
  const char* env_dir = std::getenv("SAVE_DIR");
  std::string save_directory = (env_dir != nullptr) ? std::string(env_dir) : default_directory;
  
  // Make sure the directory name is ended with '/'
  if (save_directory.back() != '/') {
    save_directory += "/";
  }
  
  if (!std::filesystem::exists(save_directory)) {
      std::filesystem::create_directories(save_directory);
  }
  return save_directory + base_name + "_" + getCurrentTimestamp() + ".csv";
}


// ----------------------------------------------------------------------------
// BufferedTreeNode
// ----------------------------------------------------------------------------

std::string bufferedTreeNodeTypeToString(BufferedTreeNodeType type)
{
  switch (type) {
    case BufferedTreeNodeType::DRIVER:
      return "Driver";
    case BufferedTreeNodeType::SINK:
      return "Sink";
    case BufferedTreeNodeType::BUFFER:
      return "Buffer";
  }
}

// ----------------------------------------------------------------------------
// BufferedTree
// ----------------------------------------------------------------------------
// YL: Deep copy constructor
BufferedTree::BufferedTree(const BufferedTree& other)
{
  netId_ = other.netId_;
  net_   = other.net_;   // You may or may not want to copy the same sta::Net*.
  root_  = copyNode(other.root_);
}

void BufferedTree::setRoot(BufferedTreeNodePtr root_node)
{
  root_ = std::move(root_node);
}

BufferedTreeNodePtr BufferedTree::getRoot() const
{
  return root_;
}

//YL
sta::Net* BufferedTree::getNet() const 
{ 
  return net_; 
}

void BufferedTree::printTree() const
{
  // Print the tree in a pre-order DFS traversal.
  int node_id = 0;
  std::queue<std::pair<BufferedTreeNodePtr, int>> q; // Queue now stores a pair of node and its depth
  q.push({root_, 0}); // Root node is at depth 0

  // std::string net_name_str;
  // net_name_str = net_->getName().c_str();
  // std::cout << "Net_name: " << net_name_str << "\t";

  while (!q.empty()) {
      auto [node, depth] = q.front(); // Unpack the node and its depth
      q.pop();
      node->node_id = node_id;
      node_id++;
      for (const auto& child : node->children) {
          q.push({child, depth + 1}); // Increment depth for children
      }
}

  // Reset the queue and print the nodes.
  auto printNode = [](BufferedTreeNodePtr node, int depth) {
    std::cout << "Depth: " << depth << "\t";
    std::cout << "Node ID: " << node->node_id << "\t";
    std::cout << "Node type: " << bufferedTreeNodeTypeToString(node->node_type) << "\t";
    std::cout << "Pin name: " << node->pin_name << "\t";
    std::cout << "Location: ( " << node->x << " , " << node->y << " )\t";
    std::cout << "Input slew: " << node->input_slew << "\t";
    std::cout << "Output slew: " << node->output_slew << "\t";
    std::cout << "Input cap: " << node->input_cap << "\t";
    std::cout << "Output cap: " << node->output_cap << "\t";
    std::cout << "Max output cap: " << node->max_output_cap << "\t";
    std::cout << "Fanout: " << node->fanout << "\t";
    std::cout << "Parent: " << (node->parent ? std::to_string(node->parent->node_id) : "None") << "\t";
    std::cout << "#Children: " << node->children.size() << "\n";
  };

  q.push({root_, 0}); // Reset the queue with root node and depth 0
  while (!q.empty()) {
    auto [node, depth] = q.front(); // Unpack the node and its depth
    q.pop();
    printNode(node, depth); 
    for (const auto& child : node->children) {
        q.push({child, depth + 1}); // Increment depth for children
    }
  }
}


void BufferedTree::clear()
{
  // Disconnect all nodes from the tree.
  std::queue<BufferedTreeNodePtr> q;
  q.push(root_);
  while (!q.empty()) {
    auto node = q.front();
    q.pop();

    for (const auto& child : node->children) {
      q.push(child);
    }

    node->children.clear();
    node->parent = nullptr;
  }
  root_ = nullptr;
}


// YL: Private helper to recursively copy a node
BufferedTreeNodePtr BufferedTree::copyNode(const BufferedTreeNodePtr &node) const
{
  if (!node) {
    return nullptr;
  }

  // Allocate a new BufferedTreeNode by copying all data from 'node'
  auto new_node = std::make_shared<BufferedTreeNode>(*node);

  // Clear children in the new node so we can re-populate them
  new_node->children.clear();

  // Recursively copy children
  for (const auto &child : node->children) {
    auto new_child = copyNode(child);
    if (new_child) {
      new_child->parent = new_node;
      new_node->children.push_back(new_child);
    }
  }

  // Return newly created node
  return new_node;
}



}  // namespace rsz
