"""Pytest fixtures for Verilator simulation tests.

The sim binary (Vtop) and firmware must already be built.
Run ``just sim-firmware`` first, then invoke with ``--run-sim``::

    uv run pytest tests -v --run-sim

Without ``--run-sim`` the tests are skipped.
"""

from __future__ import annotations

import os
import select
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = REPO_ROOT / "build" / "sim"
VTOP = BUILD_DIR / "Vtop"
FW_BIN = REPO_ROOT / "zig-out" / "bin" / "firmware.bin"
SIM_PORT = 21452


def pytest_addoption(parser):
    parser.addoption(
        "--run-sim", action="store_true", default=False,
        help="run Verilator simulation tests (requires pre-built sim)",
    )


def _write_all(fd: int, data: bytes) -> None:
    while data:
        n = os.write(fd, data)
        if n == 0:
            raise OSError("write returned 0")
        data = data[n:]


def _wait_for_banner(
    sim_fd: int, sim_proc: subprocess.Popen[bytes], timeout: float,
) -> None:
    banner = b"[link] ready\n"
    buf = b""
    deadline = time.time() + timeout
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError("timed out waiting for sim boot banner")
        r, _, _ = select.select([sim_fd], [], [], min(remaining, 1.0))
        if sim_fd in r:
            chunk = os.read(sim_fd, 4096)
            if not chunk:
                raise RuntimeError("sim stdout closed before boot banner")
            buf += chunk
            if banner in buf:
                return
            if len(buf) > len(banner):
                buf = buf[-(len(banner) - 1):]
        if sim_proc.poll() is not None:
            raise RuntimeError(
                f"sim exited with code {sim_proc.returncode} before boot banner"
            )


def _kill_sim(sim: subprocess.Popen[bytes]) -> None:
    if sim.poll() is not None:
        return
    try:
        os.killpg(sim.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        sim.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(sim.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        sim.wait()


def _bridge_threads(
    conn: socket.socket, sim_in_fd: int, sim_out_fd: int, stop: threading.Event,
) -> list[threading.Thread]:

    def sim_to_tcp():
        while not stop.is_set():
            try:
                data = os.read(sim_out_fd, 4096)
            except (OSError, ValueError):
                break
            if not data:
                break
            try:
                conn.sendall(data)
            except (OSError, BrokenPipeError):
                break

    def tcp_to_sim():
        while not stop.is_set():
            try:
                data = conn.recv(4096)
            except (OSError, ConnectionResetError, socket.timeout):
                break
            if not data:
                break
            try:
                _write_all(sim_in_fd, data)
            except (OSError, BrokenPipeError):
                break
        stop.set()

    t1 = threading.Thread(target=sim_to_tcp, daemon=True, name="sim→tcp")
    t2 = threading.Thread(target=tcp_to_sim, daemon=True, name="tcp→sim")
    t1.start()
    t2.start()
    return [t1, t2]


@pytest.fixture(scope="session")
def sim_port(request: pytest.FixtureRequest) -> str:
    """Start a Verilator sim, bridge it to TCP, return ``tcp://127.0.0.1:<port>``.

    Session-scoped — the sim starts once per test session.
    """
    if not request.config.getoption("--run-sim", default=False):
        pytest.skip("needs --run-sim (and pre-built sim — run `just sim-firmware` first)")

    sim_log = open(BUILD_DIR / "sim_stderr.log", "wb")

    sim_cmd = [
        sys.executable, "-u", "-m", "soc.sim",
        "--cfu-rows", "8", "--cfu-cols", "8",
        "--cfu-store-depth", "512", "--cfu-in-width", "8",
        "--l2-size", "128",
        "--sdram-init", str(FW_BIN),
        "--output-dir", str(BUILD_DIR),
        "--non-interactive",
    ]
    sim = subprocess.Popen(
        sim_cmd,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=sim_log, start_new_session=True, bufsize=0,
    )

    try:
        assert sim.stdout is not None
        sim_out_fd = sim.stdout.fileno()
        _wait_for_banner(sim_out_fd, sim, timeout=600.0)

        assert sim.stdin is not None
        sim_in_fd = sim.stdin.fileno()

        # TCP listener — accept + bridge runs in background thread so
        # the fixture yields immediately; the test's TcpTransport.connect()
        # unblocks accept().
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", SIM_PORT))
        listener.listen(1)

        stop = threading.Event()
        conn_ref: list[socket.socket | None] = [None]
        bridge_ref: list[list[threading.Thread]] = [[]]

        def _accept_and_bridge():
            conn = listener.accept()[0]
            conn_ref[0] = conn
            bridge_ref[0] = _bridge_threads(conn, sim_in_fd, sim_out_fd, stop)

        accept_thread = threading.Thread(target=_accept_and_bridge, daemon=True)
        accept_thread.start()

        yield f"tcp://127.0.0.1:{SIM_PORT}"

    except Exception:
        _kill_sim(sim)
        raise

    # Teardown
    stop.set()
    if conn_ref[0] is not None:
        conn_ref[0].close()
    listener.close()
    for t in bridge_ref[0]:
        t.join(timeout=2)
    accept_thread.join(timeout=2)
    _kill_sim(sim)
    sim_log.close()
