#ifndef LOOM_C_API_H
#define LOOM_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LoomHandle LoomHandle;

typedef enum {
  LOOM_OK = 0,
  LOOM_INVALID_ARGUMENT = 1,
  LOOM_PAYLOAD_TOO_LARGE = 2,
  LOOM_BAD_RESPONSE = 3,
  LOOM_BAD_MAGIC = 4,
  LOOM_UNKNOWN_OP = 5,
  LOOM_BAD_PAYLOAD_LEN = 6,
  LOOM_BAD_ADDRESS = 7,
  LOOM_ILLEGAL_INSTRUCTION = 8,
  LOOM_TRAP_FAULT = 9,
  LOOM_DEVICE_ERROR = 10,
  LOOM_OUT_OF_MEMORY = 11,
  LOOM_IO_ERROR = 12,
} loom_status_t;

loom_status_t loom_open(const char *port_path, uint32_t baud_rate, LoomHandle **out_handle);
void loom_close(LoomHandle *handle);
loom_status_t loom_ping(LoomHandle *handle);
uint16_t loom_last_cycles(LoomHandle *handle);
const char *loom_status_string(loom_status_t code);

loom_status_t loom_write_mem(LoomHandle *handle, uint32_t addr, const uint8_t *data, size_t len);
loom_status_t loom_read_mem(LoomHandle *handle, uint32_t addr, uint8_t *buf, size_t len);
loom_status_t loom_exec(LoomHandle *handle, const uint8_t *program, size_t program_len, uint32_t *out_cycles);
loom_status_t loom_exec_model(LoomHandle *handle, uint32_t model_addr, uint32_t model_len, uint32_t *out_cycles);
loom_status_t loom_exec_model_bytes(
    LoomHandle *handle,
    uint32_t model_addr,
    const uint8_t *model,
    size_t model_len,
    uint32_t *out_cycles);

#ifdef __cplusplus
}
#endif

#endif
