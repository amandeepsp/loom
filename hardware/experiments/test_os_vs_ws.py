import logging
import numpy as np
import pytest
from amaranth.sim import Simulator

from hardware.systolic.os_pe_array import OutputStationaryPEArray
from hardware.systolic.ws_pe_array import WeightStationaryPEArray

logger = logging.getLogger(__name__)


class TestOSvsWS:
    """Benchmark Output-Stationary vs Weight-Stationary systolic arrays."""

    def _run_os(self, R: int, C: int, K: int, A: np.ndarray, B: np.ndarray):
        """Drive a full [R,K] x [K,C] GEMM on the OS array."""
        dut = OutputStationaryPEArray(R, C)
        result_container = {}

        async def testbench(ctx):
            cycles = 0

            ctx.set(dut.psum_load, 1)
            await ctx.tick()
            cycles += 1
            ctx.set(dut.psum_load, 0)

            for k in range(K):
                for r in range(R):
                    ctx.set(getattr(dut, f"act_in_{r}"), int(A[r, k]))
                for c in range(C):
                    ctx.set(getattr(dut, f"w_in_{c}"), int(B[k, c]))
                await ctx.tick()
                cycles += 1

            for r in range(R):
                ctx.set(getattr(dut, f"act_in_{r}"), 0)
            for c in range(C):
                ctx.set(getattr(dut, f"w_in_{c}"), 0)
            for _ in range(R + C - 2):
                await ctx.tick()
                cycles += 1

            result = np.zeros((R, C), dtype=np.int32)
            for r in range(R):
                for c in range(C):
                    result[r, c] = ctx.get(getattr(dut, f"psum_out_{r}_{c}"))

            result_container["cycles"] = cycles
            result_container["result"] = result

        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()
        return result_container["cycles"], result_container["result"]

    def _run_ws(self, R: int, C: int, K: int, A: np.ndarray, B: np.ndarray):
        """Drive a full [R,K] x [K,C] GEMM on the WS array (row-at-a-time)."""
        assert K % R == 0
        dut = WeightStationaryPEArray(R, C)
        num_k_tiles = K // R
        result_container = {}

        async def testbench(ctx):
            cycles = 0
            accum = np.zeros((R, C), dtype=np.int32)

            for kt in range(num_k_tiles):
                k_start = kt * R

                ctx.set(dut.w_load, 1)
                for r in reversed(range(R)):
                    for c in range(C):
                        ctx.set(getattr(dut, f"w_in_{c}"), int(B[k_start + r, c]))
                    await ctx.tick()
                    cycles += 1
                ctx.set(dut.w_load, 0)

                for m in range(R):
                    for r in range(R):
                        ctx.set(getattr(dut, f"act_in_{r}"), int(A[m, k_start + r]))
                    await ctx.tick()
                    cycles += 1

                    for r in range(R):
                        ctx.set(getattr(dut, f"act_in_{r}"), 0)

                    for _ in range(R - 1):
                        await ctx.tick()
                        cycles += 1
                    accum[m, 0] += ctx.get(dut.psum_out_0)

                    for c in range(1, C):
                        await ctx.tick()
                        cycles += 1
                        accum[m, c] += ctx.get(getattr(dut, f"psum_out_{c}"))

            result_container["cycles"] = cycles
            result_container["result"] = accum

        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()
        return result_container["cycles"], result_container["result"]

    def _run_os_tiled(
        self, R: int, C: int, M: int, K: int, k_tile: int, A: np.ndarray, B: np.ndarray
    ):
        """Tiled [M,K] x [K,C] GEMM on OS array with sequential epilogue drain."""
        assert M % R == 0 and K % k_tile == 0
        dut = OutputStationaryPEArray(R, C)
        num_m_tiles = M // R
        num_k_tiles = K // k_tile
        result = np.zeros((M, C), dtype=np.int32)
        result_container = {}

        async def testbench(ctx):
            cycles = 0

            for mt in range(num_m_tiles):
                m_base = mt * R

                ctx.set(dut.psum_load, 1)
                await ctx.tick()
                cycles += 1
                ctx.set(dut.psum_load, 0)

                for kt in range(num_k_tiles):
                    k_base = kt * k_tile
                    for k in range(k_tile):
                        for r in range(R):
                            ctx.set(
                                getattr(dut, f"act_in_{r}"),
                                int(A[m_base + r, k_base + k]),
                            )
                        for c in range(C):
                            ctx.set(
                                getattr(dut, f"w_in_{c}"),
                                int(B[k_base + k, c]),
                            )
                        await ctx.tick()
                        cycles += 1

                    for r in range(R):
                        ctx.set(getattr(dut, f"act_in_{r}"), 0)
                    for c in range(C):
                        ctx.set(getattr(dut, f"w_in_{c}"), 0)
                    for _ in range(R + C - 2):
                        await ctx.tick()
                        cycles += 1

                for r in range(R):
                    for c in range(C):
                        result[m_base + r, c] = ctx.get(
                            getattr(dut, f"psum_out_{r}_{c}")
                        )
                        await ctx.tick()
                        cycles += 1

            result_container["cycles"] = cycles
            result_container["result"] = result

        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()
        return result_container["cycles"], result_container["result"]

    def _run_ws_tiled(
        self, R: int, C: int, M: int, K: int, A: np.ndarray, B: np.ndarray
    ):
        """Tiled [M,K] x [K,C] GEMM on WS array (row-at-a-time)."""
        assert K % R == 0
        dut = WeightStationaryPEArray(R, C)
        num_k_tiles = K // R
        result = np.zeros((M, C), dtype=np.int32)
        result_container = {}

        async def testbench(ctx):
            cycles = 0

            for kt in range(num_k_tiles):
                k_start = kt * R

                ctx.set(dut.w_load, 1)
                for r in reversed(range(R)):
                    for c in range(C):
                        ctx.set(getattr(dut, f"w_in_{c}"), int(B[k_start + r, c]))
                    await ctx.tick()
                    cycles += 1
                ctx.set(dut.w_load, 0)

                for m in range(M):
                    for r in range(R):
                        ctx.set(getattr(dut, f"act_in_{r}"), int(A[m, k_start + r]))
                    await ctx.tick()
                    cycles += 1

                    for r in range(R):
                        ctx.set(getattr(dut, f"act_in_{r}"), 0)
                    for _ in range(R - 1):
                        await ctx.tick()
                        cycles += 1
                    result[m, 0] += ctx.get(dut.psum_out_0)

                    for c in range(1, C):
                        await ctx.tick()
                        cycles += 1
                        result[m, c] += ctx.get(getattr(dut, f"psum_out_{c}"))

            result_container["cycles"] = cycles
            result_container["result"] = result

        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()
        return result_container["cycles"], result_container["result"]

    def _run_ws_streaming(
        self, R: int, C: int, M: int, K: int, A: np.ndarray, B: np.ndarray
    ):
        """
        WS array with continuous activation streaming (no idle cycles).
        Rows are fed back-to-back; results read as they emerge.

        Timing verified against test_ws_pe_array.py::test_4x4_gemm:
          - Row m fed at cycle N produces col c at cycle N + (R-1) + c.
        """
        assert K % R == 0
        dut = WeightStationaryPEArray(R, C)
        num_k_tiles = K // R
        result = np.zeros((M, C), dtype=np.int32)
        result_container = {}

        async def testbench(ctx):
            cycles = 0

            for kt in range(num_k_tiles):
                k_start = kt * R

                # Load weights (bottom row first)
                ctx.set(dut.w_load, 1)
                for r in reversed(range(R)):
                    for c in range(C):
                        ctx.set(getattr(dut, f"w_in_{c}"), int(B[k_start + r, c]))
                    await ctx.tick()
                    cycles += 1
                ctx.set(dut.w_load, 0)

                # Compute: M + R + C - 1 cycles
                # Row m fed at cycle R + m.
                # Row m col c ready at cycle R + m + R + c = 2R + m + c.
                # At iteration t, overall cycle = R + t.
                # Reading before tick at cycle R + t gives value computed at cycle R + t - 1.
                # We need R + t - 1 = 2R + m + c  =>  t - 1 = R + m + c  =>  m = t - R - c - 1?
                #
                # Let's verify with existing test_4x4_gemm:
                #   Row m fed at cycle N. After ticks N,N+1,N+2,N+3, read at cycle N+4.
                #   At cycle N+4 (before tick), ctx.get() reads value from tick N+3.
                #   So result computed at cycle feed_m + R - 1 = N + 3.
                #   Ready to read at cycle feed_m + R = N + 4.
                #
                # In our loop, t = 0 corresponds to cycle R (feed row 0).
                # We read at the START of each iteration, before tick.
                # At iteration t, cycle = R + t.
                # We want to read row m col c when cycle = feed_m + R + c = R + m + R + c.
                # So R + t = R + m + R + c  =>  t = m + R + c  =>  m = t - R - c.
                for t in range(M + R + C - 1):
                    # Read results that became available after the previous tick
                    for c in range(C):
                        m = t - R - c
                        if 0 <= m < M:
                            if c == 0:
                                result[m, c] += ctx.get(dut.psum_out_0)
                            else:
                                result[m, c] += ctx.get(getattr(dut, f"psum_out_{c}"))

                    # Feed next row if available
                    if t < M:
                        for r in range(R):
                            ctx.set(
                                getattr(dut, f"act_in_{r}"),
                                int(A[t, k_start + r]),
                            )
                    else:
                        for r in range(R):
                            ctx.set(getattr(dut, f"act_in_{r}"), 0)

                    await ctx.tick()
                    cycles += 1

            result_container["cycles"] = cycles
            result_container["result"] = result

        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()
        return result_container["cycles"], result_container["result"]

    # ------------------------------------------------------------------
    # 1. Single-tile benchmark
    # ------------------------------------------------------------------

    def test_os_vs_ws_benchmark(self):
        """Single-tile benchmark: identical int8 GEMMs on OS and WS arrays."""
        configs = [
            (4, 4), (4, 8), (4, 16), (4, 32), (4, 64), (4, 128),
            (8, 8), (8, 16), (8, 32), (8, 64), (8, 128), (8, 256),
        ]

        lines = [
            "",
            "=" * 90,
            "OS vs WS — Single-Tile Benchmark (M = R, array-level, no epilogue)",
            "=" * 90,
            f"{'R×C':>6} {'K':>6} {'OS cycles':>12} {'OS theory':>12} "
            f"{'WS cycles':>12} {'WS theory':>12} {'Winner':>8} {'Ratio':>8}",
            "-" * 90,
        ]

        for R, K in configs:
            C = R
            rng = np.random.default_rng(seed=42 + R * 1000 + K)
            A = rng.integers(-8, 8, size=(R, K), dtype=np.int8)
            B = rng.integers(-8, 8, size=(K, C), dtype=np.int8)
            expected = A.astype(np.int32) @ B.astype(np.int32)

            os_cycles, os_result = self._run_os(R, C, K, A, B)
            ws_cycles, ws_result = self._run_ws(R, C, K, A, B)

            np.testing.assert_array_equal(os_result, expected)
            np.testing.assert_array_equal(ws_result, expected)

            os_theory = 1 + K + (R + C - 2)
            ws_theory = (K // R) * (R + R * (1 + (R - 1) + (C - 1)))
            winner = "OS" if os_cycles < ws_cycles else "WS" if ws_cycles < os_cycles else "Tie"
            ratio = ws_cycles / os_cycles if os_cycles > 0 else float("inf")

            lines.append(
                f"{R}x{C:>3} {K:>6} {os_cycles:>12} {os_theory:>12} "
                f"{ws_cycles:>12} {ws_theory:>12} {winner:>8} {ratio:>8.2f}x"
            )

        lines += [
            "-" * 90,
            "Notes:",
            "  - OS theory = 1 (psum_load) + K (feed) + (R+C-2) (flush)",
            "  - WS theory = K/R * (R load + R rows * (1 feed + R-1 wait + C-1 read))",
            "  - A real-system OS adds R*C cycles per tile for sequential epilogue readback.",
            "=" * 90,
        ]
        logger.info("\n".join(lines))

    # ------------------------------------------------------------------
    # 2. Large-M benchmark (row-at-a-time WS)
    # ------------------------------------------------------------------

    def test_os_vs_ws_large_m_benchmark(self):
        configs = [
            (4, 4, 16, 8, 4),
            (4, 4, 16, 16, 4),
            (4, 4, 32, 8, 4),
            (8, 8, 16, 8, 8),
            (8, 8, 16, 16, 8),
            (8, 8, 32, 8, 8),
            (8, 8, 32, 64, 8),
        ]

        lines = [
            "",
            "=" * 110,
            "OS vs WS — Large-M Benchmark (M > R, tiled, with OS epilogue drain)",
            "=" * 110,
            f"{'R×C':>6} {'M':>6} {'K':>6} {'OS cycles':>12} {'OS theory':>12} "
            f"{'WS cycles':>12} {'WS theory':>12} {'Winner':>8} {'Ratio':>8}",
            "-" * 110,
        ]

        for R, C, M, K, k_tile in configs:
            rng = np.random.default_rng(seed=43 + R * 1000 + M * 100 + K)
            A = rng.integers(-8, 8, size=(M, K), dtype=np.int8)
            B = rng.integers(-8, 8, size=(K, C), dtype=np.int8)
            expected = A.astype(np.int32) @ B.astype(np.int32)

            os_cycles, os_result = self._run_os_tiled(R, C, M, K, k_tile, A, B)
            ws_cycles, ws_result = self._run_ws_tiled(R, C, M, K, A, B)

            np.testing.assert_array_equal(os_result, expected)
            np.testing.assert_array_equal(ws_result, expected)

            num_m_tiles = M // R
            num_k_tiles = K // k_tile
            os_per_m_tile = (
                1
                + (num_k_tiles - 1) * (k_tile + R + C - 2)
                + (k_tile + R + C - 2 + R * C)
            )
            os_theory = num_m_tiles * os_per_m_tile
            ws_theory = (K // R) * (R + M * (R + C))

            winner = "OS" if os_cycles < ws_cycles else "WS" if ws_cycles < os_cycles else "Tie"
            ratio = ws_cycles / os_cycles if os_cycles > 0 else float("inf")

            lines.append(
                f"{R}x{C:>3} {M:>6} {K:>6} {os_cycles:>12} {os_theory:>12} "
                f"{ws_cycles:>12} {ws_theory:>12} {winner:>8} {ratio:>8.2f}x"
            )

        lines += [
            "-" * 110,
            "Notes:",
            "  - OS theory = M/R * [1 + (K/k_tile-1)*(k_tile+R+C-2) + (k_tile+R+C-2+R*C)]",
            "  - WS theory = K/R * [R + M*(R+C)]",
            "  - Epilogue drain = R*C sequential readback cycles per OS M-tile.",
            "=" * 110,
        ]
        logger.info("\n".join(lines))

    # ------------------------------------------------------------------
    # 3. Streaming WS benchmark (continuous activation feed)
    # ------------------------------------------------------------------

    def test_os_vs_ws_streaming_benchmark(self):
        configs = [
            (4, 4, 8, 4),
            (4, 4, 16, 4),
            (4, 4, 16, 8),
            (4, 4, 32, 8),
            (8, 8, 8, 8),
            (8, 8, 16, 8),
            (8, 8, 32, 8),
            (8, 8, 64, 8),
            (8, 8, 8, 16),
            (8, 8, 16, 16),
            (8, 8, 32, 16),
            (8, 8, 64, 64),
        ]

        lines = [
            "",
            "=" * 115,
            "OS vs WS — Continuous Streaming Benchmark (no idle cycles between rows)",
            "=" * 115,
            f"{'R×C':>6} {'M':>6} {'K':>6} {'OS cycles':>12} {'OS theory':>12} "
            f"{'WS str':>12} {'WS theory':>12} {'Winner':>8} {'Ratio':>8}",
            "-" * 115,
        ]

        for R, C, M, K in configs:
            rng = np.random.default_rng(seed=44 + R * 1000 + M * 100 + K)
            A = rng.integers(-8, 8, size=(M, K), dtype=np.int8)
            B = rng.integers(-8, 8, size=(K, C), dtype=np.int8)
            expected = A.astype(np.int32) @ B.astype(np.int32)

            os_cycles, os_result = self._run_os_tiled(R, C, M, K, R, A, B)
            ws_cycles, ws_result = self._run_ws_streaming(R, C, M, K, A, B)

            np.testing.assert_array_equal(os_result, expected)
            np.testing.assert_array_equal(ws_result, expected)

            num_m_tiles = M // R
            num_k_tiles = K // R
            os_per_m_tile = (
                1
                + (num_k_tiles - 1) * (R + R + C - 2)
                + (R + R + C - 2 + R * C)
            )
            os_theory = num_m_tiles * os_per_m_tile
            ws_theory = (K // R) * (R + M + (R - 1) + (C - 1))

            winner = (
                "OS"
                if os_cycles < ws_cycles
                else "WS"
                if ws_cycles < os_cycles
                else "Tie"
            )
            ratio = ws_cycles / os_cycles if os_cycles > 0 else float("inf")

            lines.append(
                f"{R}x{C:>3} {M:>6} {K:>6} {os_cycles:>12} {os_theory:>12} "
                f"{ws_cycles:>12} {ws_theory:>12} {winner:>8} {ratio:>8.2f}x"
            )

        lines += [
            "-" * 115,
            "Notes:",
            "  - OS theory = M/R * [1 + (K/R-1)*(2R+C-2) + (2R+C-2+R*C)]  (k_tile=R)",
            "  - WS theory = K/R * [R + M + R-1 + C-1]  =  K/R * [M + 2R + C - 2]",
            "  - WS streaming eliminates per-row idle wait; pipeline stays full.",
            "  - Crossover: WS wins when K/R * (M + 2R + C - 2) < OS_total",
            "=" * 115,
        ]
        logger.info("\n".join(lines))
