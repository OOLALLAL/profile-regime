import argparse
import logging
from pathlib import Path
from dataclasses import dataclass

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

@dataclass
class Args:
    bars_1m_dir: Path
    out_dir: Path
    tf: str
    overwrite: bool
    log_level: str

    # r normalization
    r_norm: str              # "range" or "vol_tanh"
    vol_window: int          # used for vol_tanh
    eps: float

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bars-1m-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tf", type=str, default="15min", help="e.g. 5min, 15min, 1h")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log-level", default="INFO")

    # r normalization choice
    p.add_argument("--r-norm", type=str, default="range", choices=["range", "vol_tanh"])
    p.add_argument("--vol-window", type=int, default=96, help="rolling window for vol_tanh (in TF bars)")
    p.add_argument("--eps", type=float, default=1e-12)

    ns = p.parse_args()


    return Args(
        bars_1m_dir=ns.bars_1m_dir,
        out_dir=ns.out_dir,
        tf=ns.tf,
        overwrite=ns.overwrite,
        log_level=ns.log_level,
        r_norm=ns.r_norm,
        vol_window=ns.vol_window,
        eps=ns.eps,
    )
def resample_1m_to_tf(df_1m: pd.DataFrame, tf: str) -> pd.DataFrame:

    df_1m = df_1m.set_index("ts")
    g = df_1m.resample(tf)

    out = pd.DataFrame(
        {
            "open": g["open"].first(),
            "high": g["high"].max(),
            "low": g["low"].min(),
            "close": g["close"].last(),
            "qty": g["qty"].sum(),
            "qty_delta": g["qty_delta"].sum(),
            "cv": g["cv"].sum(),
            "cvd": g["cvd"].sum(),
            "trade_count": g["trade_count"].sum(),
        }
    )
    out = out.dropna().reset_index()
    return out

def compute_features(df: pd.DataFrame, r_norm: str, vol_window: int, eps: float) -> pd.DataFrame:

    df["r"] = df["close"] / df["open"] - 1.0

    df["ncvd"] = df["cvd"] / (df["cv"] + eps)
    # df["ncvd"].clip(-1.0, 1.0)

    if r_norm == "range":
        rng = df["high"] - df["low"]
        df["r_star"] = (df["close"] - df["open"]) / (rng + eps)
    elif r_norm == "vol_tanh":
        sigma = df["r"].rolling(vol_window, min_periods=max(10, vol_window // 5)).std()
        z = df["r"] / (sigma + eps)
        df["r_star"] = np.tanh(z)
    else:
        raise ValueError(f"Unknown r_norm={r_norm}")

    return df

def write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(out_path)

def run(args: Args) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)

    bars_files = sorted(args.bars_1m_dir.glob("*.parquet"))
    if not bars_files:
        raise SystemExit(f"No 1m bar parquet files found")

    for bp in bars_files:
        logger.info("PROCESS %s", bp.name)

        name = bp.name
        feat_path = args.out_dir / f"{name.replace('bars-1m', f'features-{args.tf}')}"

        if parquet_exists(feat_path) and not args.overwrite:
            logger.info("SKIP (exists) %s", name)
            continue

        df_1m = pd.read_parquet(bp)
        df_tf = resample_1m_to_tf(df_1m, args.tf)
        df_tf = compute_features(df_tf, args.r_norm, args.vol_window, args.eps)
        write_parquet(df_tf, feat_path)

        logger.info(
            "OK %s: %d bars",
            feat_path, len(df_tf),
        )

if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.log_level)
    run(args)