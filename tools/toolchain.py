"""Pinned FPGA toolchain: check it, snapshot it, and fingerprint bitstreams.

The Tang Nano 20K design sits at ~92% of the LUT sites, so only some nextpnr
seeds place (about 1 in 3). A given seed is deterministic for a fixed netlist
and toolchain, so a bitstream is reproducible only if the toolchain is too.
``soc/toolchain.json`` records it; ``soc/bitstream.json`` records the flashed
bitstream (seed, top.v hash, .fs hash, utilization, Fmax).

    uv run python -m tools.toolchain check       # exit 1 on any mismatch
    uv run python -m tools.toolchain snapshot    # print the current toolchain
    uv run python -m tools.toolchain snapshot --write   # re-pin (deliberately)
    uv run python -m tools.toolchain manifest --build-dir build/sipeed_tang_nano_20k --seed 9
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOCK = REPO / "soc" / "toolchain.json"
LITEX_ROOT = REPO.parent / "litex"
# Path-sourced in pyproject.toml ([tool.uv.sources]); everything else is
# pinned by uv.lock.
LITEX_REPOS = [
    "litex", "litex-boards", "litedram", "liteeth", "pythondata-cpu-vexriscv",
    "pythondata-software-compiler_rt", "pythondata-software-picolibc",
    "pythondata-misc-tapcfg",
]
BOARDS_PATCH = REPO / "soc" / "patches" / "litex-boards-sipeed_tang_nano_20k.patch"


def oss_cad_bin() -> Path:
    return Path(os.environ.get("OSS_CAD_BIN", Path.home() / "Projects/oss-cad-suite/bin"))


def _run(cmd: list[str], **kw) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    except FileNotFoundError:
        return ""
    return (r.stdout + r.stderr).strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot() -> dict:
    cad = oss_cad_bin()
    version_file = cad.parent / "VERSION"
    yosys = re.search(r"Yosys (\S+) \(git sha1 (\w+)", _run([str(cad / "yosys"), "-V"]))
    nextpnr = re.search(r"Version (\S+)\)", _run([str(cad / "nextpnr-himbaechel"), "--version"]))
    apycula = sorted(cad.parent.glob("lib/python3*/site-packages/Apycula-*.egg-info")) or \
        sorted(cad.parent.glob("lib/python3*/site-packages/apycula-*.dist-info"))
    zig = _run(["zig", "version"])
    repos = {}
    for name in LITEX_REPOS:
        d = LITEX_ROOT / name
        repos[name] = {
            "commit": _run(["git", "-C", str(d), "rev-parse", "HEAD"]),
            "url": _run(["git", "-C", str(d), "remote", "get-url", "origin"]),
        }
    return {
        "oss_cad_suite": version_file.read_text().strip() if version_file.exists() else "",
        "yosys": f"{yosys.group(1)} {yosys.group(2)}" if yosys else "",
        "nextpnr_himbaechel": nextpnr.group(1) if nextpnr else "",
        "apycula": re.sub(r"^[Aa]pycula-|\.(egg|dist)-info$|-py3\.\d+", "", apycula[-1].name) if apycula else "",
        "zig": zig,
        "litex_repos": repos,
        "litex_boards_patch": {
            "path": str(BOARDS_PATCH.relative_to(REPO)),
            "sha256": sha256(BOARDS_PATCH),
            "applied": patch_applied(),
        },
    }


def patch_applied() -> bool:
    """The patch is applied iff it reverse-applies cleanly to litex-boards."""
    r = subprocess.run(
        ["git", "-C", str(LITEX_ROOT / "litex-boards"), "apply", "--check", "-R", str(BOARDS_PATCH)],
        capture_output=True)
    return r.returncode == 0


def diff(want: dict, have: dict, prefix: str = "") -> list[str]:
    out = []
    for k, w in want.items():
        h = have.get(k)
        if isinstance(w, dict):
            out += diff(w, h if isinstance(h, dict) else {}, f"{prefix}{k}.")
        elif k != "url" and w != h:
            out.append(f"  {prefix}{k}: pinned {w!r}, found {h!r}")
    return out


def cmd_check(_args) -> int:
    want = json.loads(LOCK.read_text())
    want.pop("_comment", None)
    problems = diff(want, snapshot())
    if not problems:
        print(f"toolchain matches {LOCK.relative_to(REPO)}")
        return 0
    print(f"toolchain differs from {LOCK.relative_to(REPO)}:", file=sys.stderr)
    print("\n".join(problems), file=sys.stderr)
    print("A bitstream built now will not reproduce the recorded seed/placement.\n"
          "Fix the toolchain, or re-pin deliberately with "
          "`uv run python -m tools.toolchain snapshot --write` (then re-screen seeds).\n"
          "Set LOOM_SKIP_TOOLCHAIN_CHECK=1 to build anyway.", file=sys.stderr)
    return 0 if os.environ.get("LOOM_SKIP_TOOLCHAIN_CHECK") else 1


def cmd_snapshot(args) -> int:
    snap = {"_comment": "Pinned FPGA toolchain; see tools/toolchain.py. "
                        "Re-pin only deliberately: a new toolchain re-rolls which pnr seeds place."}
    snap.update(snapshot())
    text = json.dumps(snap, indent=2) + "\n"
    if args.write:
        LOCK.write_text(text)
        print(f"wrote {LOCK.relative_to(REPO)}")
    else:
        print(text, end="")
    return 0


def cmd_manifest(args) -> int:
    build = Path(args.build_dir)
    gw = build / "gateware"
    fs = gw / "sipeed_tang_nano_20k.fs"
    report = gw / "sipeed_tang_nano_20k.report"
    top = Path(args.top)
    man: dict = {
        "seed": args.seed,
        "git": _run(["git", "-C", str(REPO), "describe", "--always", "--dirty", "--abbrev=12"]),
        "top_v_sha256": sha256(top),
        "fs_sha256": sha256(fs) if fs.exists() else None,
    }
    marker = build / "loom_cfu_features"
    if marker.exists():
        man["cfu_features"] = marker.read_text().strip()
    if report.exists():
        rep = json.loads(report.read_text())
        u = rep.get("utilization", {})
        pick = lambda k: u.get(k, {}).get("used")  # noqa: E731
        man["utilization"] = {k: pick(k) for k in ("LUT4", "ALU", "DFF", "RAM16SDP4", "BSRAM", "MULT9X9")}
        if pick("LUT4") is not None:
            man["utilization"]["LUT4+ALU"] = pick("LUT4") + (pick("ALU") or 0)
            man["utilization"]["LUT_sites"] = u["LUT4"]["available"]
        man["fmax_mhz"] = {k: round(v["achieved"], 2) for k, v in rep.get("fmax", {}).items()}
    man["toolchain"] = snapshot()
    (build / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    print(f"wrote {build / 'manifest.json'}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    s = sub.add_parser("snapshot")
    s.add_argument("--write", action="store_true", help=f"overwrite {LOCK.relative_to(REPO)}")
    s.set_defaults(fn=cmd_snapshot)
    m = sub.add_parser("manifest", help="fingerprint a built bitstream")
    m.add_argument("--build-dir", required=True)
    m.add_argument("--seed", type=int, required=True)
    m.add_argument("--top", default=str(REPO / "top.v"))
    m.set_defaults(fn=cmd_manifest)
    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
