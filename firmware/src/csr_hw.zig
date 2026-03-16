// LiteX CSR addresses for sipeed_tang_nano_20k SoC (VexRiscv_FullCfu @ 0xf0000000).
//
// Regenerate from: build/sipeed_tang_nano_20k/software/include/generated/csr.h
//   grep '#define CSR_.*_ADDR' csr.h  (then lowercase, strip prefix/suffix)

pub const CSR_BASE: usize = 0xf0000000;

// -- ctrl --
pub const ctrl_reset: usize = 0xf0000800;
pub const ctrl_scratch: usize = 0xf0000804;
pub const ctrl_bus_errors: usize = 0xf0000808;

// -- timer0 --
pub const timer0_load: usize = 0xf0002800;
pub const timer0_reload: usize = 0xf0002804;
pub const timer0_en: usize = 0xf0002808;
pub const timer0_update_value: usize = 0xf000280c;
pub const timer0_value: usize = 0xf0002810;
pub const timer0_ev_status: usize = 0xf0002814;
pub const timer0_ev_pending: usize = 0xf0002818;
pub const timer0_ev_enable: usize = 0xf000281c;

// -- uart --
pub const uart_rxtx: usize = 0xf0003000;
pub const uart_txfull: usize = 0xf0003004;
pub const uart_rxempty: usize = 0xf0003008;
pub const uart_ev_status: usize = 0xf000300c;
pub const uart_ev_pending: usize = 0xf0003010;
pub const uart_ev_enable: usize = 0xf0003014;
pub const uart_txempty: usize = 0xf0003018;
pub const uart_rxfull: usize = 0xf000301c;
