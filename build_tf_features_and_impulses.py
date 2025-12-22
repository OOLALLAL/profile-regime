import argparse
import logging
from pathlib import Path
import dataclasses as dataclass

import numpy as np
import pandas as pd

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

    # impulse thresholds (quantiles)
    q_vol: float
    q_ncvd: float
    q_align: float
    q_low_vol: float
    q_low_ncvd: float
    min_impulse_bars: int

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

    # impulse quantiles
    p.add_argument("--q-vol", type=float, default=0.85, help="quantile for |r*|")
    p.add_argument("--q-ncvd", type=float, default=0.60, help="quantile for |cvd/cv|")
    p.add_argument("--q-align", type=float, default=0.50, help="quantile for x=r* * cvd/cv")
    p.add_argument("--q-low-vol", type=float, default=0.25, help="quantile for |r*| end condition")
    p.add_argument("--q-low-ncvd", type=float, default=0.30, help="quantile for |cvd/cv| end condition")
    p.add_argument("--min-impulse-bars", type=int, default=2, help="min TF bars per impulse event")

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
        q_vol=ns.q_vol,
        q_ncvd=ns.q_ncvd,
        q_align=ns.q_align,
        q_low_vol=ns.q_low_vol,
        q_low_ncvd=ns.q_low_ncvd,
        min_impulse_bars=ns.min_impulse_bars,
    )
def resample_1m_to_tf(df_1m: pd.DataFrame, tf: str) -> pd.DataFrame:

    df_1m = df_1m.copy().set_index("ts")
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
            "trade_count": g["trade_count"].count(),
        }
    )
    out.dropna().reset_index()
    return out
def compute_features(df: pd.DataFrame, r_norm: str, vol_window: int, eps: float) -> pd.DataFrame:
    df = df.copy()

    df["r"] = df["close"] / df["open"] - 1.0

    df["ncvd"] = df["cvd"] / (df["cv"] + eps)
    # df["ncvd"].clip(-1.0, 1.0)

    if r_norm == "range":
        rng = df["high"] - df["low"]
        df["r_star"] = df["r"] / (rng + eps)
    elif r_norm == "vol_tanh":
        sigma = df["r"].rolling(vol_window, min_periods=max(10, vol_window // 5)).std()
        z = df["r"] / (sigma + eps)
        df["r_star"] = np.tanh(z)
    else:
        raise ValueError(f"Unknown r_norm={r_norm}")

    df["x"] = df["r"] * df["ncvd"]
    return df

def _quantile_thresholds(df: pd.DataFrame, q_vol: float, q_flow: float, q_align: float, q_low_vol: float, q_low_flow: float) -> Dict[str, float]:
    abs_r = df["r_star"].abs()
    abs_ncvd = df["ncvd"].abs()
    x = df["x"]

    th = {
        "abs_r_hi": float(abs_r.quantile(q_vol)),
    }

def run(args: Args) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)

    bars_files = sorted(args.bars_1m_dir.glob("*.parquet"))
    if not bars_files:
        raise SystemExit(f"No 1m bar parquet files found")

    features_dir = args.out_dir / "features"
    impulses_dir = args.out_dir / "impulses"

    for bp in bars_files:
        logger.info("PROCESS $s", bp.name)

        name = bp.name
        feat_path = features_dir / f"{name.replace('bars_1m', f'features_{args.tf}')}"
        imp_path = impulses_dir / f"{name.replace('bars_1m', f'impulses_{args.tf}')}"

        if parquet_exists(feat_path) and parquet_exists(imp_path):
            logger.info("SKIP (exists) %s", name)
            continue

        df_1m = pd.read_parquet(bp)
        df_tf = resample_1m_to_tf(df_1m, args.tf)
        df_tf = compute_features(df_tf, args.r_norm, args.vol_window, args.eps)

        th = _quantile_thresholds(df_tf, args.q_vol, args.q_ncvd, args.q_align, args.q_low_vol, args.q_low_ncvd)
        events = detect_impulse(df_tf, th, args.min_impulse_bars)

if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.log_level)
    run(args)