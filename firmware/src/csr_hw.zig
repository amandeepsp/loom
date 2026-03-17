// Auto-generated from build/sipeed_tang_nano_20k/csr.json — do not edit.
// Regenerate: python firmware/gen_csr.py build/sipeed_tang_nano_20k/csr.json

pub const CSR_BASE: usize = 0xf0000000;

// -- buttons --
pub const buttons_in: usize = 0xf0000000;

// -- ctrl --
pub const ctrl_reset: usize = 0xf0000800;
pub const ctrl_scratch: usize = 0xf0000804;
pub const ctrl_bus_errors: usize = 0xf0000808;

// -- leds --
pub const leds_out: usize = 0xf0001800;

// -- sdram --
pub const sdram_dfii_control: usize = 0xf0002000;
pub const sdram_dfii_pi0_command: usize = 0xf0002004;
pub const sdram_dfii_pi0_command_issue: usize = 0xf0002008;
pub const sdram_dfii_pi0_address: usize = 0xf000200c;
pub const sdram_dfii_pi0_baddress: usize = 0xf0002010;
pub const sdram_dfii_pi0_wrdata: usize = 0xf0002014;
pub const sdram_dfii_pi0_rddata: usize = 0xf0002018;

// -- spiflash --
pub const spiflash_phy_clk_divisor: usize = 0xf0002800;
pub const spiflash_mmap_dummy_bits: usize = 0xf0002804;
pub const spiflash_master_cs: usize = 0xf0002808;
pub const spiflash_master_phyconfig: usize = 0xf000280c;
pub const spiflash_master_rxtx: usize = 0xf0002810;
pub const spiflash_master_status: usize = 0xf0002814;

// -- timer0 --
pub const timer0_load: usize = 0xf0003000;
pub const timer0_reload: usize = 0xf0003004;
pub const timer0_en: usize = 0xf0003008;
pub const timer0_update_value: usize = 0xf000300c;
pub const timer0_value: usize = 0xf0003010;
pub const timer0_ev_status: usize = 0xf0003014;
pub const timer0_ev_pending: usize = 0xf0003018;
pub const timer0_ev_enable: usize = 0xf000301c;

// -- uart --
pub const uart_rxtx: usize = 0xf0003800;
pub const uart_txfull: usize = 0xf0003804;
pub const uart_rxempty: usize = 0xf0003808;
pub const uart_ev_status: usize = 0xf000380c;
pub const uart_ev_pending: usize = 0xf0003810;
pub const uart_ev_enable: usize = 0xf0003814;
pub const uart_txempty: usize = 0xf0003818;
pub const uart_rxfull: usize = 0xf000381c;
