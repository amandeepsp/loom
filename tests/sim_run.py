"""Run a command against a Verilator sim, bridging TCP <-> sim's UART (stdio).

LiteX's `serial2tcp` simulation module is broken on macOS (libevent/kqueue
interaction never delivers UART bytes to the TCP client). Workaround: spawn
`soc.sim` with `serial2console` (default), capture its stdin/stdout pipes,
and bridge them to a TCP socket on the requested port.

Layout while running:
    inner cmd  <--TCP-->  sim_run  <--pipes-->  soc.sim (Vsim)

The inner command (e.g. tools.test_gemm) connects to 127.0.0.1:<port> and
talks the wire protocol. sim_run forwards every byte verbatim in both
directions.

Usage:
    python -m tools.sim_run --port 21450 --sim-arg ... -- <cmd> <args...>
"""

from __future__ import annotations

import argparse
import contextlib
import os
import select
import signal
import socket
import subprocess
import sys
import threading
import time


def _write_all(fd: int, data: bytes) -> None:
    """Write all bytes to fd, handling partial writes from os.write()."""
    while data:
        n = os.write(fd, data)
        if n == 0:
            raise OSError("write returned 0")
        data = data[n:]


def _bridge(src_read, dst_write, stop: threading.Event, log_file=None) -> None:
    """Pump bytes from src_read() into dst_write() until EOF or stop event."""
    while not stop.is_set():
        try:
            data = src_read(4096)
        except (OSError, ValueError):
            break
        if not data:
            break
        if log_file is not None:
            log_file.write(data)
            log_file.flush()
        try:
            dst_write(data)
        except (OSError, BrokenPipeError):
            break
    stop.set()


def _wait_for_banner(
    fd: int,
    sim_process: subprocess.Popen[bytes],
    timeout: float,
) -> None:
    """Read from sim stdout until the firmware boot banner appears.

    The firmware writes "[link] ready\\n" after uart.drainRx() and before
    entering its main receive loop.  Consuming this banner guarantees the
    firmware has finished booting and won't drainRx() away any request bytes.
    """
    banner = b"[link] ready\n"
    buf = b""
    deadline = time.time() + timeout
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError("timed out waiting for sim boot banner")
        r, _, _ = select.select([fd], [], [], min(remaining, 1.0))
        if fd in r:
            chunk = os.read(fd, 4096)
            if not chunk:
                raise RuntimeError("sim stdout closed before boot banner")
            buf += chunk
            if banner in buf:
                return
            # Keep overlap region for banner spanning chunk boundaries.
            if len(buf) > len(banner):
                buf = buf[-(len(banner) - 1):]
        if sim_process.poll() is not None:
            raise RuntimeError(
                f"sim exited with code {sim_process.returncode} before boot banner"
            )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True,
                   help="TCP port the inner command will connect to")
    p.add_argument("--sim-arg", action="append", default=[],
                   help="Argument forwarded to soc.sim (repeatable)")
    p.add_argument("--sim-log", default="/tmp/sim.log",
                   help="File for sim build/runtime logs (stderr)")
    p.add_argument("--uart-log", default="/tmp/sim_uart.log",
                   help="File for sim UART stdout bytes")
    p.add_argument("--accept-timeout", type=float, default=120.0,
                   help="Seconds to wait for inner cmd to connect")
    p.add_argument("--boot-timeout", type=float, default=600.0,
                   help="Seconds to wait for firmware boot banner on sim stdout")
    p.add_argument("cmd", nargs=argparse.REMAINDER,
                   help="Command to run (prefix with --)")
    args = p.parse_args()

    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        print("sim_run: no command provided", file=sys.stderr)
        return 2

    # Open TCP listener BEFORE spawning anything, so the port is reserved.
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", args.port))
    listener.listen(1)

    sim_log = open(args.sim_log, "wb")
    uart_log = open(args.uart_log, "wb")

    # Spawn sim with stdio piped. Own process group so we can kill the whole
    # tree (soc.sim -> bash run_sim.sh -> Vsim) on exit.
    sim_cmd = [sys.executable, "-u", "-m", "soc.sim", *args.sim_arg, "--non-interactive"]
    sim = subprocess.Popen(
        sim_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sim_log,
        start_new_session=True,
        bufsize=0,
    )

    # Wait for the firmware to boot before spawning the inner command.
    # This prevents a race: if the inner command sent requests while the sim
    # was still building, those bytes would sit buffered in the stdin pipe,
    # then get drained by firmware's uart.drainRx() on boot → deadlock.
    assert sim.stdout is not None
    sim_out_fd = sim.stdout.fileno()
    try:
        _wait_for_banner(sim_out_fd, sim, args.boot_timeout)
    except (TimeoutError, RuntimeError, OSError) as exc:
        print(f"sim_run: sim boot failed: {exc}", file=sys.stderr)
        if sim.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(sim.pid, signal.SIGTERM)
            try:
                sim.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(sim.pid, signal.SIGKILL)
                sim.wait()
        listener.close()
        sim_log.close()
        uart_log.close()
        return 1

    inner: subprocess.Popen[bytes] | None = None
    bridge_threads: list[threading.Thread] = []
    stop = threading.Event()

    def cleanup() -> None:
        stop.set()
        if inner is not None and inner.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                inner.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                inner.wait(timeout=2)
        if sim.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(sim.pid, signal.SIGTERM)
            try:
                sim.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(sim.pid, signal.SIGKILL)
                sim.wait()
        for t in bridge_threads:
            t.join(timeout=1)
        with contextlib.suppress(OSError):
            listener.close()
        sim_log.close()
        uart_log.close()

    def on_signal(signum: int, _frame: object) -> None:
        cleanup()
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGHUP, on_signal)

    try:
        # Spawn inner command — firmware is ready, banner already consumed.
        inner = subprocess.Popen(cmd)

        listener.settimeout(args.accept_timeout)
        try:
            conn, _addr = listener.accept()
        except socket.timeout:
            print(f"sim_run: inner cmd never connected to :{args.port}",
                  file=sys.stderr)
            return 1
        finally:
            listener.close()

        # Spin up bidirectional pipe<->socket bridges. Use os.read on the
        # raw fd: Popen(bufsize=0).stdout is a FileIO whose .read(n) blocks
        # until n bytes arrive — useless for line-rate UART (~50 byte boot
        # banner then idle).
        assert sim.stdin is not None
        sim_in_fd = sim.stdin.fileno()

        bridge_threads = [
            threading.Thread(
                target=_bridge,
                args=(lambda n: os.read(sim_out_fd, n), conn.sendall, stop, uart_log),
                daemon=True, name="sim->tcp",
            ),
            threading.Thread(
                target=_bridge,
                args=(conn.recv, lambda d: _write_all(sim_in_fd, d), stop),
                daemon=True, name="tcp->sim",
            ),
        ]
        for t in bridge_threads:
            t.start()

        rc = inner.wait()
        return rc
    finally:
        cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
