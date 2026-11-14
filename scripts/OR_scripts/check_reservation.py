#!/usr/bin/env python3
"""check_reservation.py <placer log> : buffer area reserved at every timing-driven trigger.

Reads the "[MLBUF] trigger" lines of a global placement log (run_replace_mlbuf_*.log, run_replace_adhoc.log)
and prints boxes read / kept, reserved area and its share of the whitespace per trigger.
Exits with status 1 if a trigger reserved no area, or if the log has no trigger line.
"""
import re
import sys

PAT = re.compile(r"\[MLBUF\] trigger (\d+): approach (\S+) boxes read (\d+) kept (\d+) "
                 r"reserved ([\d.]+) um\^2 \(([\d.]+)% of whitespace\)")


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    rows = [m.groups() for m in map(PAT.search, open(sys.argv[1], errors="ignore")) if m]
    if not rows:
        sys.exit(f"no [MLBUF] trigger line in {sys.argv[1]}")
    print(f"{'trigger':>7} {'approach':>8} {'read':>6} {'kept':>6} {'reserved um^2':>14} {'% whitespace':>13}")
    for t, approach, read, kept, area, pct in rows:
        print(f"{t:>7} {approach:>8} {read:>6} {kept:>6} {float(area):>14.1f} {float(pct):>13.3f}")
    empty = [t for t, _, _, _, area, _ in rows if float(area) <= 0.0]
    if empty:
        sys.exit(f"FAIL: no area reserved at trigger(s) {', '.join(empty)}")
    print(f"OK: area reserved at all {len(rows)} triggers")


if __name__ == "__main__":
    main()
