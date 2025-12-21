import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

def parquet_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


@dataclass
class Args:
    trades_dir: Path
    out_dir: Path
    overwrite: bool
    log_level: str

def parse_args() -> Args:
    p = argparse.ArgumentParser("trade parquet -> 1m bar parquet")
    p.add_argument("--trades-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log-level", default="INFO")
    ns = p.parse_args()

    return Args(
        trades_dir=ns.trades_dir,
        out_dir=ns.out_dir,
        overwrite=ns.overwrite,
        log_level=ns.log_level,
    )

def build_1m_bar(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["transact_time"], unit="ms", utc=True)
    df = df.assign(ts=dt).set_index("ts")

    g = df.resample("1min")

    bar = pd.DataFrame(
        {
        "open": g["price"].first(),
        "high": g["price"].max(),
        "low": g["price"].min(),
        "close": g["price"].last(),
        "qty": g["quantity"].sum(),
        "qty_delta": g["signed_quantity"].sum(),
        "cv": g["volume"].sum(),
        "cvd": g["signed_volume"].sum(),
        "trade_count": g["price"].count(),
        }
    )
    bar = bar.dropna().reset_index()
    return bar

def run(args: Args) -> None:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    trades = sorted(args.trades_dir.glob("*.parquet"))
    if not trades:
        raise SystemExit("No trade parquet files found")

    for tp in trades:
        logger.info("PROCESS %s", tp.name)

        df = pd.read_parquet(tp)
        bar = build_1m_bar(df)

        out_path = out_dir / tp.name.replace("aggTrades", "bars-1m")
        if parquet_exists(out_path) and not args.overwrite:
            logger.info("SKIP %s (exists)", out_path)
            continue
        tmp = out_path.with_suffix(".parquet.part")

        table = pa.Table.from_pandas(bar, preserve_index=False)
        pq.write_table(table, tmp, compression="zstd")
        tmp.replace(out_path)

        logger.info("OK %s (%d bars)", out_path.name, len(bar))

if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.log_level)
    run(args)
