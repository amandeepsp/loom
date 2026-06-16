from pathlib import Path


def test_c_api_declares_model_execution_entrypoints():
    header = Path("host/c_api.h").read_text()
    source = Path("host/c_api.zig").read_text()

    for name in ("loom_exec_model", "loom_exec_model_bytes"):
        assert f"loom_status_t {name}(" in header
        assert f"pub export fn {name}(" in source

    assert "protocol_error" not in source
    assert "LOOM_BAD_PAYLOAD_LEN" in header
    assert "LOOM_TRAP_FAULT" in header


def test_host_driver_uses_zig_serial_package():
    build_zon = Path("build.zig.zon").read_text()
    build = Path("build.zig").read_text()
    driver = Path("host/driver.zig").read_text()

    assert "github.com/ZigEmbeddedGroup/serial" in build_zon
    assert 'b.dependency("serial"' in build
    assert 'driver_mod.addImport("serial"' in build
    assert 'const serial = @import("serial");' in driver
    assert "serial.configureSerialPort" in driver
    assert "serial.flushSerialPort" in driver
    assert "termios" not in driver
    assert "tcsetattr" not in driver
