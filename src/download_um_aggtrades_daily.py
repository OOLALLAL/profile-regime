import argparse
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Tuple, Optional

import requests

BASE_URL = "https://data.binance.vision"
PATH_TMPL = "/data/futures/um/daily/aggTrades/{symbol}/{symbol}-aggTrades-{ymd}.zip"
DEFAULT_TIMEOUT: Tuple[int, int] = (5, 60) # (connect, read)

logger = logging.getLogger(__name__)

def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

def parse_yyyymmdd(s: str) -> date:
    return datetime.strptime(s, "%Y%m%d").date()

def daterange_inclusive(start: date, end: date) -> Iterable[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

def download_one(url: str, out_path: Path, timeout=DEFAULT_TIMEOUT, max_retries: int = 5) -> Tuple[bool, Optional[int]]:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 0:
        logger.info("SKIP %s (already exists, %.2f MB)", out_path.name, out_path.stat().st_size / 1024 / 1024)
        return True, 200

    last_exc: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code != 200:
                logger.warning("MISS %s %s", r.status_code, url)
                return False, r.status_code

            tmp = out_path.with_suffix(out_path.suffix + ".part")
            tmp.write_bytes(r.content)
            tmp.replace(out_path)

            logger.info("OK  %s (%.2f MB)", out_path.name, out_path.stat().st_size / 1024 / 1024)
            return True, 200

        except (requests.Timeout, requests.ConnectionError) as e:
            last_exc = e
            sleep = 1.5 * attempt
            logger.warning("RETRY %d/%d: %s (sleep %.1fs)", attempt, max_retries, type(e).__name__, sleep)
            time.sleep(sleep)

    logger.error("FAIL %s (%s)", url, last_exc)
    return False, None

@dataclass
class Args:
    symbol: str
    start: date
    end: date
    out_dir: Path
    sleep_sec: float
    log_level: str

def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True, help="e.g. BTCUSDT")
    p.add_argument("--start", required=True, help="YYYYMMDD ( UTC date)")
    p.add_argument("--end", required=True, help="YYYYMMDD ( UTC date)")
    p.add_argument("--out-dir", type=Path, default=Path("D:/data/profile-regime"), help="Base output directory")
    p.add_argument("--sleep-sec", type=float, default=0.2, help="Sleep between requests to be polite")
    p.add_argument("--log_level", type=str, default="INFO", help="DEBUG|INFO|WARNING|ERROR")
    ns = p.parse_args()

    start = parse_yyyymmdd(ns.start)
    end = parse_yyyymmdd(ns.end)
    if end < start:
        raise SystemExit(f"--end must be >= --start")

    return Args(
        symbol=ns.symbol.upper(),
        start=start,
        end=end,
        out_dir=ns.out_dir,
        sleep_sec=ns.sleep_sec,
        log_level=ns.log_level,
    )

def run(args: Args) -> None:

    zips_dir = args.out_dir / "raw" / "binance_data" / "futures_um" / "daily" / "aggTrades" / f"symbol={args.symbol}"

    ok, miss = 0, 0
    for d in daterange_inclusive(args.start, args.end):
        ymd = d.isoformat()
        url = f"{BASE_URL}{PATH_TMPL.format(symbol=args.symbol, ymd=ymd)}"
        zip_name = f"{args.symbol}-aggTrades-{ymd}.zip"
        zip_path = zips_dir / zip_name

        logger.info("DOWNLOADING %s",zip_name)
        success, _ = download_one(url, zip_path)
        if success:
            ok += 1
        else:
            miss += 1

        time.sleep(args.sleep_sec)

    logger.info("DONE | downloaded=%d missing=%d | saved_to=%s", ok, miss, zips_dir)

if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.log_level)

    run(args)
