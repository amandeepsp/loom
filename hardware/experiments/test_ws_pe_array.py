from amaranth.sim import Simulator

from hardware.systolic.ws_pe_array import WeightStationaryPEArray


class TestWeightStationaryPEArray:
    def test_2x2_gemm(self):
        """
        C = A @ B on WS array. Same data as OS test.
          A = [[1, 2],   B = [[5, 6],
               [3, 4]]        [7, 8]]
          C = [[19, 22],
               [43, 50]]

        B is loaded as weights (stationary). Each row of A is fed as
        activations — one M-row per pass, weights stay loaded.
        Results stagger: col c exits at tick R-1+c after feed.
        """

        async def testbench(ctx):
            A = [[1, 2], [3, 4]]
            B = [[5, 6], [7, 8]]
            expected = [[19, 22], [43, 50]]

            # Load B as weights, bottom row first
            ctx.set(dut.w_load, 1)
            ctx.set(dut.w_in_0, B[1][0])
            ctx.set(dut.w_in_1, B[1][1])
            await ctx.tick()
            ctx.set(dut.w_in_0, B[0][0])
            ctx.set(dut.w_in_1, B[0][1])
            await ctx.tick()
            ctx.set(dut.w_load, 0)

            # Process each M-row
            for m in range(2):
                ctx.set(dut.act_in_0, A[m][0])
                ctx.set(dut.act_in_1, A[m][1])
                await ctx.tick()
                ctx.set(dut.act_in_0, 0)
                ctx.set(dut.act_in_1, 0)

                await ctx.tick()
                assert ctx.get(dut.psum_out_0) == expected[m][0], \
                    f"C[{m}][0] = {ctx.get(dut.psum_out_0)}, expected {expected[m][0]}"
                await ctx.tick()
                assert ctx.get(dut.psum_out_1) == expected[m][1], \
                    f"C[{m}][1] = {ctx.get(dut.psum_out_1)}, expected {expected[m][1]}"

        dut = WeightStationaryPEArray(2, 2)
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_ws_array_2x2.vcd"):
            sim.run()

    def test_2x2_gemm_negatives(self):
        """
        Signed GEMM, same data as OS test.
          A = [[ 1, -2],   B = [[-3,  4],
               [ 5,  6]]        [ 7, -8]]
          C = [[-17, 20],
               [27, -28]]
        """

        async def testbench(ctx):
            A = [[1, -2], [5, 6]]
            B = [[-3, 4], [7, -8]]
            expected = [[-17, 20], [27, -28]]

            ctx.set(dut.w_load, 1)
            ctx.set(dut.w_in_0, B[1][0])
            ctx.set(dut.w_in_1, B[1][1])
            await ctx.tick()
            ctx.set(dut.w_in_0, B[0][0])
            ctx.set(dut.w_in_1, B[0][1])
            await ctx.tick()
            ctx.set(dut.w_load, 0)

            for m in range(2):
                ctx.set(dut.act_in_0, A[m][0])
                ctx.set(dut.act_in_1, A[m][1])
                await ctx.tick()
                ctx.set(dut.act_in_0, 0)
                ctx.set(dut.act_in_1, 0)

                await ctx.tick()
                assert ctx.get(dut.psum_out_0) == expected[m][0]
                await ctx.tick()
                assert ctx.get(dut.psum_out_1) == expected[m][1]

        dut = WeightStationaryPEArray(2, 2)
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_ws_array_neg.vcd"):
            sim.run()

    def test_4x4_gemm(self):
        """
        4x4 GEMM, same data as OS test.
          A = [[1, 0, 2, -1],    B = [[ 1, 2, 0, -1],
               [0, 3, -1, 0],         [ 0, 1, 3,  2],
               [2, 1, 0, 4],          [-1, 0, 1,  0],
               [-1, 0, 1, 2]]         [ 2, -1, 0,  3]]
          C = [[-3, 3, 2, -4],
               [ 1, 3, 8,  6],
               [10, 1, 3, 12],
               [ 2, -4, 1, 7]]

        B loaded as weights once. Each A row fed as a separate M-pass.
        """

        async def testbench(ctx):
            A = [[ 1, 0, 2,-1],
                 [ 0, 3,-1, 0],
                 [ 2, 1, 0, 4],
                 [-1, 0, 1, 2]]
            B = [[ 1, 2, 0,-1],
                 [ 0, 1, 3, 2],
                 [-1, 0, 1, 0],
                 [ 2,-1, 0, 3]]
            expected = [[-3, 3, 2,-4],
                        [ 1, 3, 8, 6],
                        [10, 1, 3,12],
                        [ 2,-4, 1, 7]]

            # Load B as weights, bottom row first
            ctx.set(dut.w_load, 1)
            for r in reversed(range(4)):
                for c in range(4):
                    ctx.set(getattr(dut, f"w_in_{c}"), B[r][c])
                await ctx.tick()
            ctx.set(dut.w_load, 0)

            for m in range(4):
                # Feed activation row m
                for r in range(4):
                    ctx.set(getattr(dut, f"act_in_{r}"), A[m][r])
                await ctx.tick()
                for r in range(4):
                    ctx.set(getattr(dut, f"act_in_{r}"), 0)

                # Col 0 exits after R-1 = 3 more ticks
                for _ in range(3):
                    await ctx.tick()
                assert ctx.get(dut.psum_out_0) == expected[m][0], \
                    f"C[{m}][0] = {ctx.get(dut.psum_out_0)}, expected {expected[m][0]}"

                for c_idx in range(1, 4):
                    await ctx.tick()
                    val = ctx.get(getattr(dut, f"psum_out_{c_idx}"))
                    assert val == expected[m][c_idx], \
                        f"C[{m}][{c_idx}] = {val}, expected {expected[m][c_idx]}"

        dut = WeightStationaryPEArray(4, 4)
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_ws_array_4x4.vcd"):
            sim.run()

    def test_2x2_k_tiled_gemm(self):
        """
        K-tiled GEMM via 2 passes, externally accumulated.
          A = [[1, 2, 3, 4],    W = [[ 1, -1], [ 2,  0], [-1,  3], [ 0,  2]]
          A @ W = [[ 2, 16], [10, 32]]
        """

        async def testbench(ctx):
            A = [[1, 2, 3, 4],
                 [5, 6, 7, 8]]
            W = [[ 1, -1],
                 [ 2,  0],
                 [-1,  3],
                 [ 0,  2]]
            expected = [[2, 16], [10, 32]]

            for m in range(2):
                accum = [0, 0]

                # K-tile 0: W[0:2]
                ctx.set(dut.w_load, 1)
                ctx.set(dut.w_in_0, W[1][0])
                ctx.set(dut.w_in_1, W[1][1])
                await ctx.tick()
                ctx.set(dut.w_in_0, W[0][0])
                ctx.set(dut.w_in_1, W[0][1])
                await ctx.tick()
                ctx.set(dut.w_load, 0)

                ctx.set(dut.act_in_0, A[m][0])
                ctx.set(dut.act_in_1, A[m][1])
                await ctx.tick()
                ctx.set(dut.act_in_0, 0)
                ctx.set(dut.act_in_1, 0)
                await ctx.tick()
                accum[0] += ctx.get(dut.psum_out_0)
                await ctx.tick()
                accum[1] += ctx.get(dut.psum_out_1)

                # K-tile 1: W[2:4]
                ctx.set(dut.w_load, 1)
                ctx.set(dut.w_in_0, W[3][0])
                ctx.set(dut.w_in_1, W[3][1])
                await ctx.tick()
                ctx.set(dut.w_in_0, W[2][0])
                ctx.set(dut.w_in_1, W[2][1])
                await ctx.tick()
                ctx.set(dut.w_load, 0)

                ctx.set(dut.act_in_0, A[m][2])
                ctx.set(dut.act_in_1, A[m][3])
                await ctx.tick()
                ctx.set(dut.act_in_0, 0)
                ctx.set(dut.act_in_1, 0)
                await ctx.tick()
                accum[0] += ctx.get(dut.psum_out_0)
                await ctx.tick()
                accum[1] += ctx.get(dut.psum_out_1)

                assert accum[0] == expected[m][0], \
                    f"C[{m}][0] = {accum[0]}, expected {expected[m][0]}"
                assert accum[1] == expected[m][1], \
                    f"C[{m}][1] = {accum[1]}, expected {expected[m][1]}"

        dut = WeightStationaryPEArray(2, 2)
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_ws_array_k_tiled.vcd"):
            sim.run()

    def test_1x1_multiply(self):
        """1x1: no skew, each cycle outputs act * w."""

        async def testbench(ctx):
            ctx.set(dut.w_load, 1)
            ctx.set(dut.w_in_0, 3)
            await ctx.tick()
            ctx.set(dut.w_load, 0)

            ctx.set(dut.act_in_0, 4)
            await ctx.tick()
            assert ctx.get(dut.psum_out_0) == 12

            ctx.set(dut.act_in_0, 2)
            await ctx.tick()
            assert ctx.get(dut.psum_out_0) == 6

        dut = WeightStationaryPEArray(1, 1)
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_ws_array_1x1.vcd"):
            sim.run()
