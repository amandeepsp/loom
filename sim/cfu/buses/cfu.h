// Patched cfu.h — adds the signal pointer fields that the /opt/renode
// v1.16.1 header is missing.  Force-included via -include so its guard
// (#define CFU_H) blocks the system copy when library sources compile.
//
// Original: Copyright (c) 2025 Antmicro — MIT License

#ifndef CFU_H
#define CFU_H
#include <cstdint>

#ifndef DEFAULT_TIMEOUT
#define DEFAULT_TIMEOUT 2000
#endif

struct Cfu
{
    virtual void tick(bool countEnable, uint64_t steps);
    virtual void reset();
    uint64_t execute(uint32_t functionID, uint32_t data0, uint32_t data1, int* error);
    void timeoutTick(uint8_t* signal, uint8_t expectedValue, int timeout);
    void (*evaluateModel)();
    uint64_t tickCounter;

    // Signal pointers — wired to the Verilated model in sim_main.cpp Init().
    uint8_t  *clk;
    uint8_t  *rst;
    uint8_t  *req_valid;
    uint8_t  *req_ready;
    uint16_t *req_func_id;
    uint32_t *req_data0;
    uint32_t *req_data1;
    uint8_t  *resp_valid;
    uint8_t  *resp_ready;
    uint32_t *resp_data;
};
#endif
