"""Reproducible Tang Nano 20K bitstream builds and nextpnr seed screening.

The design fills ~92% of the GW2AR-18's LUT sites, so only some nextpnr seeds
place. Placement is deterministic for a given netlist, so a pinned seed is
reproducible only if the netlist is. Two things made LiteX netlists differ
build to build, and both are removed here:

* the build timestamp in the SoC identifier ROM (a LUT ROM) and the BIOS
  (``--no-ident-version`` + a fixed ``SOURCE_DATE_EPOCH``);
* absolute source paths in the Yosys script, which Yosys bakes into internal
  cell names (rewritten relative to the gateware dir; every build dir sits at
  ``build/<name>/gateware``, so the strings match across dirs).

With both, two builds in different dirs give byte-identical netlists, so a
seed screened in ``build/seeds`` places the same in the real build.

    python -m tools.soc_build screen --seeds "1 2 3" -- <litex args>   # place only
    python -m tools.soc_build build --seed 9 -- <litex args>           # full bitstream
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

NAME = "sipeed_tang_nano_20k"
TARGET = f"litex_boards.targets.{NAME}"
DEVICE = ["--device", "GW2AR-LV18QN88C8/I7", "--vopt", "family=GW2A-18C", "--vopt", f"cst={NAME}.cst"]
SOURCE_DATE_EPOCH = "1767225600"  # 2026-01-01; any constant works


def generate(out: Path, soc_args: list[str]) -> Path:
    """LiteX sources + BIOS (the ROM init is part of the netlist), then a
    Yosys script with relative paths. Returns the gateware dir."""
    env = dict(os.environ, SOURCE_DATE_EPOCH=SOURCE_DATE_EPOCH)
    subprocess.run([sys.executable, "-m", TARGET, "--build", "--no-compile-gateware",
                    "--no-ident-version", "--output-dir", str(out), *soc_args], check=True, env=env)
    gw = out / "gateware"
    ys = gw / f"{NAME}.ys"
    text = re.sub(r"^(read_verilog\s+)(\S+)$",
                  lambda m: m.group(1) + os.path.relpath(m.group(2), gw.resolve()),
                  ys.read_text(), flags=re.M)
    ys.write_text(text)
    return gw


def synth(gw: Path) -> None:
    print("synthesizing...", flush=True)
    subprocess.run(["yosys", "-q", "-l", f"{NAME}.rpt", f"{NAME}.ys"], cwd=gw, check=True)


def place(gw: Path, seed: int, threads: int) -> tuple[int, str, float]:
    log = gw.parent / f"place_seed{seed}.log"
    t0 = time.time()
    with log.open("w") as f:
        rc = subprocess.run(
            ["nextpnr-himbaechel", "--json", f"{NAME}.json", "--top", NAME, *DEVICE,
             "--timing-allow-fail", "--no-route", "--threads", str(threads), "--seed", str(seed)],
            cwd=gw, stdout=f, stderr=subprocess.STDOUT).returncode
    if "Unable to find legal placement" in log.read_text(errors="replace"):
        status = "fail"
    elif rc == 0:
        status = "PLACED"
    else:
        status = f"error rc={rc} (see {log})"
    return seed, status, time.time() - t0


def cmd_screen(args, soc_args: list[str]) -> int:
    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    gw = generate(Path(args.output_dir), soc_args)
    synth(gw)
    print(f"placing {len(seeds)} seeds, {args.jobs} at a time (~3 min each)...", flush=True)
    placed = []
    with ThreadPoolExecutor(args.jobs) as ex:
        for seed, status, dt in ex.map(lambda s: place(gw, s, args.threads), seeds):
            print(f"  seed {seed:3}: {status}  ({dt:.0f}s)", flush=True)
            if status == "PLACED":
                placed.append(seed)
    print(f"\n{len(placed)}/{len(seeds)} seeds placed: {' '.join(map(str, placed)) or 'none'}")
    if placed:
        print(f"pin one as pnr_seed in the Justfile, then: just soc  (or just soc seed={placed[0]})")
    return 0 if placed else 1


def cmd_build(args, soc_args: list[str]) -> int:
    gw = generate(Path(args.output_dir), soc_args)
    synth(gw)
    subprocess.run(["nextpnr-himbaechel", "--json", f"{NAME}.json", "--top", NAME, *DEVICE,
                    "--report", f"{NAME}.report", "--write", f"{NAME}_routed.json",
                    "--timing-allow-fail", "--seed", str(args.seed)], cwd=gw, check=True)
    subprocess.run(["gowin_pack", "-d", "GW2A-18C", "-o", f"{NAME}.fs", f"{NAME}_routed.json",
                    "--mspi_as_gpio", "--sspi_as_gpio", "--ready_as_gpio", "--done_as_gpio"],
                   cwd=gw, check=True)
    print(f"wrote {gw / (NAME + '.fs')}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("screen", help="placement-only seed screen")
    s.add_argument("--seeds", default=" ".join(map(str, range(1, 17))))
    s.add_argument("--jobs", type=int, default=16)
    s.add_argument("--threads", type=int, default=2, help="nextpnr threads per seed")
    s.add_argument("--output-dir", default="build/seeds")
    s.set_defaults(fn=cmd_screen)
    b = sub.add_parser("build", help="full bitstream at one seed")
    b.add_argument("--seed", type=int, required=True)
    b.add_argument("--output-dir", default=f"build/{NAME}")
    b.set_defaults(fn=cmd_build)
    args, rest = p.parse_known_args()
    return args.fn(args, [a for a in rest if a != "--"])


if __name__ == "__main__":
    sys.exit(main())
