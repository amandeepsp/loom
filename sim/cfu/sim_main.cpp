// sim_main.cpp — Bridges the Verilated CFU model to Renode.
//
// Follows the CFU-Playground pattern: signal wiring only, all bus
// protocol is handled by the integration library's Cfu::execute().
//
// VCD tracing: set ACCEL_VCD=<path> to dump waveforms, e.g.
//   export ACCEL_VCD=cfu_trace.vcd
//
// Reference: github.com/google/CFU-Playground/blob/main/
//            common/renode-verilator-integration/sim_main.cpp

#include <csignal>
#include <cstdlib>
#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vtop.h"

#include "renode_cfu.h"
#include "buses/cfu.h"

Vtop *top = new Vtop;
VerilatedVcdC *tfp = nullptr;
uint64_t sim_time = 0;

void vcd_cleanup() {
    if (tfp) { tfp->flush(); tfp->close(); tfp = nullptr; }
}

void sig_handler(int) { vcd_cleanup(); _exit(0); }

void eval() {
    top->eval();
    if (tfp) {
        tfp->dump(sim_time);
        sim_time++;
    }
}

RenodeAgent *Init() {
    // Enable VCD tracing if ACCEL_VCD is set.
    const char *vcd_path = std::getenv("ACCEL_VCD");
    if (vcd_path) {
        Verilated::traceEverOn(true);
        tfp = new VerilatedVcdC;
        top->trace(tfp, 99);  // 99 levels of hierarchy
        tfp->open(vcd_path);
        std::atexit(vcd_cleanup);
        std::signal(SIGTERM, sig_handler);
        std::signal(SIGINT, sig_handler);
    }

    Cfu *bus = new Cfu();

    bus->req_valid    = &top->cmd_valid;
    bus->req_ready    = &top->cmd_ready;
    bus->req_func_id  = (uint16_t *)&top->cmd_payload_function_id;
    bus->req_data0    = (uint32_t *)&top->cmd_payload_inputs_0;
    bus->req_data1    = (uint32_t *)&top->cmd_payload_inputs_1;
    bus->resp_valid   = &top->rsp_valid;
    bus->resp_ready   = &top->rsp_ready;
    bus->resp_data    = (uint32_t *)&top->rsp_payload_outputs_0;
    bus->rst          = &top->reset;
    bus->clk          = &top->clk;

    bus->evaluateModel = &eval;

    return new RenodeAgent(bus);
}
