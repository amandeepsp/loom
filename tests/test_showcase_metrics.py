from __future__ import annotations

import math

from tools.showcase_metrics import _ceil_tile, _sweep_rows


def test_ceil_tile():
    assert _ceil_tile(1) == 8
    assert _ceil_tile(8) == 8
    assert _ceil_tile(9) == 16


def test_sweep_rows_compute_theoretical_and_achieved_values():
    sweep = {
        "results": [
            {"m": 8, "k": 8, "n": 8, "variant": "non-pipelined", "cycles": 128},
            {
                "m": 8,
                "k": 8,
                "n": 8,
                "variant": "pipelined",
                "cycles": 64,
                "counters": {
                    "seq_busy_cycles": 16,
                    "dma_active_cycles": 12,
                    "overlap_cycles": 4,
                },
            },
        ]
    }
    roofline, overlap = _sweep_rows(sweep, clock_hz=48_000_000, peak_gops=6.144)

    operations = 2 * 8 * 8 * 8
    minimum_bytes = 8 * 8 + 8 * 8 + 8 * 8 + 12 * 8
    assert len(roofline) == len(overlap) == 1
    assert math.isclose(roofline[0]["arithmetic_intensity"], operations / minimum_bytes)
    assert math.isclose(roofline[0]["streamed_kir_gops"], operations * 48_000_000 / 64 / 1e9)
    assert math.isclose(roofline[0]["array_active_gops"], operations * 48_000_000 / 16 / 1e9)
    assert math.isclose(
        roofline[0]["accelerator_window_gops"],
        operations * 48_000_000 / (16 + 12 - 4) / 1e9,
    )
    assert overlap[0]["overlap_speedup"] == 2.0
    assert overlap[0]["cycles_saved"] == 64
