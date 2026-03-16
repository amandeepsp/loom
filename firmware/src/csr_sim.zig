// LiteX CSR addresses for sim SoC (VexRiscv_FullCfu @ 0xf0000000).
//
// Regenerate from: build/sim/software/include/generated/csr.h
//   grep '#define CSR_.*_ADDR' csr.h  (then lowercase, strip prefix/suffix)

pub const CSR_BASE: usize = 0xf0000000;

// -- ctrl --
pub const ctrl_reset: usize = 0xf0000000;
pub const ctrl_scratch: usize = 0xf0000004;
pub const ctrl_bus_errors: usize = 0xf0000008;

// -- timer0 --
pub const timer0_load: usize = 0xf0001000;
pub const timer0_reload: usize = 0xf0001004;
pub const timer0_en: usize = 0xf0001008;
pub const timer0_update_value: usize = 0xf000100c;
pub const timer0_value: usize = 0xf0001010;
pub const timer0_ev_status: usize = 0xf0001014;
pub const timer0_ev_pending: usize = 0xf0001018;
pub const timer0_ev_enable: usize = 0xf000101c;

// -- uart --
pub const uart_rxtx: usize = 0xf0001800;
pub const uart_txfull: usize = 0xf0001804;
pub const uart_rxempty: usize = 0xf0001808;
pub const uart_ev_status: usize = 0xf000180c;
pub const uart_ev_pending: usize = 0xf0001810;
pub const uart_ev_enable: usize = 0xf0001814;
pub const uart_txempty: usize = 0xf0001818;
pub const uart_rxfull: usize = 0xf000181c;
