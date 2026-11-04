"""Board facts the host needs to turn cycle counts into time."""

from __future__ import annotations

import json
from pathlib import Path

HW_CSR = Path(__file__).resolve().parent.parent / "build/sipeed_tang_nano_20k/csr.json"
FALLBACK_CLOCK_HZ = 48_000_000  # the board's nominal crystal-derived max


def clock_hz(csr_json: Path | str = HW_CSR) -> int:
    """SoC clock from LiteX's csr.json (``config_clock_frequency``).

    The bitstream runs at whatever LiteX was built for (40.5 MHz now, not the
    48 MHz the board name suggests), so ms/GOPS must come from here. Falls back
    to 48 MHz only when no csr.json exists yet.
    """
    try:
        consts = json.loads(Path(csr_json).read_text())["constants"]
        return int(consts["config_clock_frequency"])
    except (OSError, KeyError, ValueError):
        return FALLBACK_CLOCK_HZ
