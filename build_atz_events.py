import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
import re

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


def parquet_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0

def write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(out_path)


FNAME_RE = re.compile(
    r"^(?P<symbol>[A-Z0-9]+)-features-(?P<tf>[^-]+)-(?P<date>\d{4}-\d{2}-\d{2})\.parquet$"
)

def parse_feature_filename(name: str) -> dict:
    m = FNAME_RE.match(name)
    if not m:
        raise ValueError(f"Unexpected feature filename: {name}")
    return m.groupdict()

def compute_rolling_thresholds(
        s: pd.Series,
        window: int,
        q: float,
        min_periods: int | None = None,
        eps: float = 1e-12,
) -> pd.Series:
    if min_periods is None:
        min_periods = max(10, window // 2)


    th = s.rolling(window, min_periods=min_periods).quantile(q)
    th = th.shift(1)
    return th + eps

def make_atz_flag(
        df: pd.DataFrame,
        window: int,
        q: float,
        use_trade_count: bool = True,
        use_cv: bool = True,
) -> pd.Series:

    flags = []

    if use_cv:
        cv_th = compute_rolling_thresholds(df["cv"], window=window, q=q)
        flags.append(df["cv"] > cv_th)
    if use_trade_count:
        tc_th = compute_rolling_thresholds(df["trade_count"], window=window, q=q)
        flags.append(df["trade_count"] > tc_th)
    if not flags:
        raise ValueError("At least one of use_cv/use_trade_count must be True")

    out = flags[0]
    for f in flags[1:]:
        out = out | f

    return out.fillna(False)

def build_events_from_flag(
    df: pd.DataFrame,
    flag: pd.Series,
    min_bars: int = 2,
    merge_gap: int = 0,
) -> pd.DataFrame:

    assert len(df) == len(flag)

    if merge_gap > 0:
        f = flag.to_numpy(dtype=bool)
        false_run = 0
        for i in range(len(f)):
            if not f[i]:
                false_run += 1
            else:
                if 0 < false_run <= merge_gap:
                    f[i - false_run: i] = True
                false_run = 0
        flag = pd.Series(f, index=flag.index)

    events = []
    in_event = False
    start = 0
    eid = 0
    for i, is_on in enumerate(flag.to_numpy(dtype=bool)):
        if is_on and not in_event:
            in_event = True
            start = i
        elif (not is_on) and in_event:
            end = i - 1
            bars = end - start + 1
            if bars >= min_bars:
                eid += 1
                seg = df.iloc[start:end+1]
                events.append(
                    {
                        "atz_id": eid,
                        "start_idx": int(start),
                        "end_idx": int(end),
                        "start_ts": seg["ts"].iloc[0],
                        "end_ts": seg["ts"].iloc[-1],
                        "n_bars": int(bars),

                        # activity stats
                        "cv_sum": float(seg["cv"].sum()),
                        "cv_mean": float(seg["cv"].mean()),
                        "trade_count_sum": float(seg["trade_count"].sum()),
                        "trade_count_mean": float(seg["trade_count"].mean()),

                        # price context
                        "open": float(seg["open"].iloc[0]),
                        "close": float(seg["close"].iloc[-1]),
                        "high": float(seg["high"].max()),
                        "low": float(seg["low"].min()),
                        "range": float(seg["high"].max() - seg["low"].min()),
                        "range_per_bar": float((seg["high"].max() - seg["low"].min()) / bars),
                        "ret": float(seg["close"].iloc[-1] / seg["open"].iloc[0] - 1.0),
                    }
                )
            in_event = False
    # if ended while in_event
    if in_event:
        end = len(df) - 1
        bars = end - start + 1
        if bars >= min_bars:
            eid += 1
            seg = df.iloc[start:end+1]
            events.append(
                {
                    "atz_id": eid,
                    "start_idx": int(start),
                    "end_idx": int(end),
                    "start_ts": seg["ts"].iloc[0],
                    "end_ts": seg["ts"].iloc[-1],
                    "n_bars": int(bars),
                    "cv_sum": float(seg["cv"].sum()),
                    "cv_mean": float(seg["cv"].mean()),
                    "trade_count_sum": float(seg["trade_count"].sum()),
                    "trade_count_mean": float(seg["trade_count"].mean()),
                    "open": float(seg["open"].iloc[0]),
                    "close": float(seg["close"].iloc[-1]),
                    "high": float(seg["high"].max()),
                    "low": float(seg["low"].min()),
                    "range": float(seg["high"].max() - seg["low"].min()),
                    "range_per_bar": float((seg["high"].max() - seg["low"].min()) / bars),
                    "ret": float(seg["close"].iloc[-1] / seg["open"].iloc[0] - 1.0),
                }
            )
    return pd.DataFrame(events)

@dataclass
class Args:
    features_dir: Path
    out_dir: Path
    symbol: str
    tf: str

    window: int
    q: float
    min_bars: int
    merge_gap: int

    use_cv: bool
    use_trade_count: bool

    overwrite: bool
    log_level: str

def parse_args():
    p = argparse.ArgumentParser("features_tf -> ATZ events")
    p.add_argument("--features-dir", type=Path, required=True,)
    p.add_argument("--out-dir", type=Path, required=True)

    p.add_argument("--symbol", type=str, required=True, help="optional filter")
    p.add_argument("--tf", type=str, required=True, help="optional filter, like 15min")

    p.add_argument("--window", type=int, default=96, help="rolling window in TF bars (96=1 day for 15m)")
    p.add_argument("--q", type=float, default=0.90, help="quantile for activity threshold")

    p.add_argument("--min-bars", type=int, default=2, help="min consecutive bars for an ATZ event")
    p.add_argument("--merge-gap", type=int, default=1, help="merge events separated by <= gap bars")

    p.add_argument("--no-cv", action="store_true", help="disable cv condition")
    p.add_argument("--no-trade-count", action="store_true", help="disable trade_count condition")

    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log-level", default="INFO")

    ns = p.parse_args()

    use_cv = True
    use_trade_count = True
    if ns.no_cv:
        use_cv = False
    if ns.no_trade_count:
        use_trade_count = False

    return Args(
        features_dir=ns.features_dir,
        out_dir=ns.out_dir,
        symbol=ns.symbol,
        tf=ns.tf,
        window=ns.window,
        q=ns.q,
        min_bars=ns.min_bars,
        merge_gap=ns.merge_gap,
        use_cv=use_cv,
        use_trade_count=use_trade_count,
        overwrite=ns.overwrite,
        log_level=ns.log_level,
    )
def run(args: Args) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fps = sorted(args.features_dir.glob("*.parquet"))
    if not fps:
        raise SystemExit(f"No feature parquet files found in {args.features_dir}")

    for fp in fps:
        meta = parse_feature_filename(fp.name)

        if meta["symbol"] != args.symbol: continue
        if meta["tf"] != args.tf: continue

        out_name = fp.name.replace("-features-", "-atz-")
        out_path = args.out_dir / out_name

        if parquet_exists(out_path) and not args.overwrite:
            logger.info("SKIP (exists) %s", out_path.name)
            continue

        logger.info("PROCESS %s", fp.name)
        df = pd.read_parquet(fp)

        # Ensure ts exists and sorted
        if "ts" not in df.columns:
            raise ValueError(f"'ts' column not found in {fp.name}")
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.sort_values("ts").reset_index(drop=True)

        # sanity columns
        for c in ["cv", "trade_count", "open", "high", "low", "close"]:
            if c not in df.columns:
                raise ValueError(f"Missing column '{c}' in {fp.name}")

        flag = make_atz_flag(
            df,
            window=args.window,
            q=args.q,
            use_trade_count=args.use_trade_count,
            use_cv=args.use_cv,
        )

        events = build_events_from_flag(
            df,
            flag,
            min_bars=args.min_bars,
            merge_gap=args.merge_gap,
        )

        events.insert(0, "tf", meta["tf"])
        events.insert(0, "date", meta["date"])
        events.insert(0, "symbol", meta["symbol"])

        write_parquet(events, out_path)
        logger.info("OK %s (%d events)", out_path.name, len(events))


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.log_level)
    run(args)