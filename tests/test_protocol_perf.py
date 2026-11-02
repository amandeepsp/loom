import struct

from shared.protocol import BaseTransport, OP_PERF, PerfCounters


class _FakeTransport(BaseTransport):
    def _request(self, op, payload, *, expected_len=None):
        assert op == OP_PERF
        assert payload == struct.pack("<I", 1)
        assert expected_len == 12
        return struct.pack("<3I", 101, 67, 23)


def test_perf_counter_wire_layout():
    counters = _FakeTransport().perf_counters(reset=True)
    assert counters == PerfCounters(
        seq_busy_cycles=101,
        dma_active_cycles=67,
        overlap_cycles=23,
    )
