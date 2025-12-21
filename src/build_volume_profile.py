import argparse
import logging
from pathlib import Path
import dataclasses as dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

def parse_utc_datetime(s: str) -> datetime:
    return datetime.fromisoformat(s)


@dataclass
class Args:
    bars_dir: Path
    out_dir: Path
    start: datetime
    end: datetime
    bin_size: float
    value_area_pct: float
    log_level: str

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bars-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--start", required=True, help='UTC start (e.g. "2025-10-01T00:00:00")')
    p.add_argument("--end", required=True, help='UTC end (e.g. "2025-10-02T00:00:00")')
    p.add_argument("--bin-size", type=float, required=True)
    p.add_argument("--value-area-pct", type=float, default=0.70)
    p.add_argument("--log-level", default="INFO")
    ns = p.parse_args()

    start = parse_utc_datetime(ns.start)
    end = parse_utc_datetime(ns.end)
    if end <= start:
        raise SystemExit("--end must be > --start")

    return Args(
        bars_dir=ns.bars_dir,
        out_dir=ns.out_dir,
        start=start,
        end=end,
        bin_size=ns.bin_size,
        value_area_pct=ns.value_area_pct,
        log_level=ns.log_level,
    )
def run(args: Args) -> None:

if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.log_level)
    run(args)