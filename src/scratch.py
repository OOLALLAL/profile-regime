import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_utc_datetime(s: str) -> datetime:
    """
    Accepts:
      - "YYYY-MM-DDTHH:MM:SS" (assumed UTC)
      - "YYYY-MM-DD HH:MM:SS" (assumed UTC)
      - "YYYY-MM-DD" (00:00:00 UTC)
    """
    s = s.strip()
    if "T" in s:
        dt = datetime.fromisoformat(s)
    elif " " in s:
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    else:
        dt = datetime.strptime(s, "%Y-%m-%d")
    return dt.replace(tzinfo=timezone.utc)


def atomic_write_parquet(df: pd.DataFrame, out_path: Path, compression: str = "zstd") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, tmp, compression=compression)
    tmp.replace(out_path)


def compute_value_area(prices: np.ndarray, vols: np.ndarray, value_area_pct: float = 0.70) -> Tuple[float, float, float]:
    """
    Returns (poc_price, val_price, vah_price)
    - POC: price bin with max volume
    - Value area: bins picked by descending volume until reaching value_area_pct of total volume
    """
    total = float(vols.sum())
    if total <= 0:
        raise ValueError("Total volume is zero; cannot compute value area.")

    poc_idx = int(np.argmax(vols))
    poc_price = float(prices[poc_idx])

    order = np.argsort(-vols)  # descending by volume
    cum = 0.0
    chosen = []
    for idx in order:
        v = float(vols[idx])
        if v <= 0:
            continue
        chosen.append(idx)
        cum += v
        if cum / total >= value_area_pct:
            break

    chosen_prices = prices[chosen]
    val_price = float(chosen_prices.min())
    vah_price = float(chosen_prices.max())
    return poc_price, val_price, vah_price


def compute_lvn(prices: np.ndarray, vols: np.ndarray) -> float:
    """
    Minimal LVN: the price bin with the smallest non-zero volume.
    (LVN has many definitions; this is a robust starting point.)
    """
    mask = vols > 0
    if not mask.any():
        raise ValueError("No non-zero bins; cannot compute LVN.")
    idx = int(np.argmin(np.where(mask, vols, np.inf)))
    return float(prices[idx])


def build_profile_from_bars(
    bars: pd.DataFrame,
    bin_size: float,
    price_col_low: str = "low",
    price_col_high: str = "high",
    vol_col: str = "cv",
    delta_col: str = "cvd",
) -> pd.DataFrame:
    """
    Build volume profile bins by distributing each bar's volume uniformly
    across the bins overlapped by [low, high].
    Uses quote volume (cv) and signed quote volume (cvd) by default.
    """
    if bars.empty:
        raise ValueError("bars is empty")

    lo = float(bars[price_col_low].min())
    hi = float(bars[price_col_high].max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError(f"Invalid price range: lo={lo}, hi={hi}")

    # Create bin edges
    start = np.floor(lo / bin_size) * bin_size
    end = np.ceil(hi / bin_size) * bin_size
    edges = np.arange(start, end + bin_size, bin_size)
    centers = (edges[:-1] + edges[1:]) / 2.0

    vol_bins = np.zeros(len(centers), dtype=np.float64)
    delta_bins = np.zeros(len(centers), dtype=np.float64)

    # Distribute each bar volume across overlapped bins
    lows = bars[price_col_low].to_numpy(dtype=np.float64)
    highs = bars[price_col_high].to_numpy(dtype=np.float64)
    vols = bars[vol_col].to_numpy(dtype=np.float64)
    deltas = bars[delta_col].to_numpy(dtype=np.float64)

    for low, high, v, dv in zip(lows, highs, vols, deltas):
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            continue
        if not np.isfinite(v) or v == 0:
            continue

        # Determine overlapped bin indices (inclusive)
        i0 = int(np.floor((low - start) / bin_size))
        i1 = int(np.floor((high - start) / bin_size))

        i0 = max(i0, 0)
        i1 = min(i1, len(centers) - 1)
        n = (i1 - i0 + 1)
        if n <= 0:
            continue

        vol_share = v / n
        delta_share = dv / n

        vol_bins[i0:i1 + 1] += vol_share
        delta_bins[i0:i1 + 1] += delta_share

    ncvd = np.divide(delta_bins, vol_bins, out=np.zeros_like(delta_bins), where=vol_bins > 0)

    prof = pd.DataFrame({
        "price": centers.astype(np.float64),
        "vol": vol_bins.astype(np.float64),
        "delta": delta_bins.astype(np.float64),
        "ncvd": ncvd.astype(np.float64),
    })

    # Drop empty bins for cleanliness
    prof = prof[prof["vol"] > 0].reset_index(drop=True)
    return prof


@dataclass
class Args:
    bars_dir: Path
    out_path: Path
    start: datetime
    end: datetime
    bin_size: float
    value_area_pct: float
    log_level: str


def parse_args() -> Args:
    p = argparse.ArgumentParser("1m bars -> fixed-window volume profile (VP)")
    p.add_argument("--bars-dir", type=Path, required=True, help="Directory containing 1m bar parquet files")
    p.add_argument("--out", type=Path, required=True, help="Output parquet path for VP bins")
    p.add_argument("--start", required=True, help='UTC start (e.g. "2025-10-01T00:00:00")')
    p.add_argument("--end", required=True, help='UTC end (e.g. "2025-10-02T00:00:00")')
    p.add_argument("--bin-size", type=float, required=True, help="Price bin size (e.g. 10, 25, 50 for BTCUSDT)")
    p.add_argument("--value-area-pct", type=float, default=0.70, help="Value area percentage (default 0.70)")
    p.add_argument("--log-level", default="INFO")
    ns = p.parse_args()

    start = parse_utc_datetime(ns.start)
    end = parse_utc_datetime(ns.end)
    if end <= start:
        raise SystemExit("--end must be > --start")

    return Args(
        bars_dir=ns.bars_dir,
        out_path=ns.out,
        start=start,
        end=end,
        bin_size=float(ns.bin_size),
        value_area_pct=float(ns.value_area_pct),
        log_level=ns.log_level,
    )


def run(args: Args) -> None:
    files = sorted(args.bars_dir.glob("*.parquet"))
    if not files:
        raise SystemExit("No parquet files found in --bars-dir")

    # Load only needed columns
    cols = ["ts", "open", "high", "low", "close", "cv", "cvd", "trade_count", "qty", "qty_delta"]
    bars_list = []
    for fp in files:
        df = pd.read_parquet(fp, columns=[c for c in cols if c in pq.read_metadata(fp).schema.to_arrow_schema().names])
        bars_list.append(df)

    bars = pd.concat(bars_list, ignore_index=True)

    # Ensure ts is datetime UTC
    if not np.issubdtype(bars["ts"].dtype, np.datetime64):
        bars["ts"] = pd.to_datetime(bars["ts"], utc=True)

    # Filter window
    bars = bars[(bars["ts"] >= args.start) & (bars["ts"] < args.end)].copy()
    if bars.empty:
        raise SystemExit("No bars in the requested window")

    logger.info("WINDOW bars=%d | %s to %s (UTC)", len(bars), args.start.isoformat(), args.end.isoformat())

    prof = build_profile_from_bars(
        bars,
        bin_size=args.bin_size,
        price_col_low="low",
        price_col_high="high",
        vol_col="cv",
        delta_col="cvd",
    )

    prices = prof["price"].to_numpy(dtype=np.float64)
    vols = prof["vol"].to_numpy(dtype=np.float64)

    poc, val, vah = compute_value_area(prices, vols, value_area_pct=args.value_area_pct)
    lvn = compute_lvn(prices, vols)

    # Add summary columns (same value repeated) for convenience
    prof["poc"] = poc
    prof["val"] = val
    prof["vah"] = vah
    prof["lvn"] = lvn

    atomic_write_parquet(prof, args.out_path, compression="zstd")

    logger.info("SAVED %s", args.out_path)
    logger.info("SUMMARY | POC=%.2f | VAL=%.2f | VAH=%.2f | LVN=%.2f | bins=%d",
                poc, val, vah, lvn, len(prof))


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.log_level)
    run(args)
