import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

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

    # impulse thresholds (quantiles)
    q_vol: float
    q_flow: float
    q_align: float
    q_low_vol: float
    q_low_flow: float
    min_impulse_bars: int


def parse_args() -> Args:
    p = argparse.ArgumentParser("1m bars -> TF bars + x feature + impulse events")

    p.add_argument("--bars-1m-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tf", type=str, default="15m", help="e.g. 5m, 15m, 1h")

    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log-level", default="INFO")

    # r normalization choice
    p.add_argument("--r-norm", type=str, default="range", choices=["range", "vol_tanh"])
    p.add_argument("--vol-window", type=int, default=96, help="rolling window for vol_tanh (in TF bars)")
    p.add_argument("--eps", type=float, default=1e-12)

    # impulse quantiles
    p.add_argument("--q-vol", type=float, default=0.85, help="quantile for |r*|")
    p.add_argument("--q-flow", type=float, default=0.60, help="quantile for |cvd/cv|")
    p.add_argument("--q-align", type=float, default=0.50, help="quantile for x=r* * cvd/cv")
    p.add_argument("--q-low-vol", type=float, default=0.25, help="quantile for |r*| end condition")
    p.add_argument("--q-low-flow", type=float, default=0.30, help="quantile for |cvd/cv| end condition")
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
        q_flow=ns.q_flow,
        q_align=ns.q_align,
        q_low_vol=ns.q_low_vol,
        q_low_flow=ns.q_low_flow,
        min_impulse_bars=ns.min_impulse_bars,
    )


# -----------------------
# 1m -> TF resample
# -----------------------
def resample_1m_to_tf(df_1m: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    Input df_1m columns expected:
      ts (datetime, UTC) or index datetime,
      open, high, low, close,
      qty, qty_delta,
      cv, cvd,
      trade_count
    """
    if "ts" in df_1m.columns:
        df_1m = df_1m.copy()
        df_1m["ts"] = pd.to_datetime(df_1m["ts"], utc=True)
        df_1m = df_1m.set_index("ts")
    else:
        df_1m = df_1m.copy()
        if df_1m.index.tz is None:
            df_1m.index = df_1m.index.tz_localize("UTC")

    g = df_1m.resample(tf)

    out = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "qty": g["qty"].sum(),
        "qty_delta": g["qty_delta"].sum(),
        "cv": g["cv"].sum(),
        "cvd": g["cvd"].sum(),
        "trade_count": g["trade_count"].sum(),
    }).dropna()

    out = out.reset_index().rename(columns={"index": "ts"})
    return out


# -----------------------
# Feature engineering
# -----------------------
def compute_features(df: pd.DataFrame, r_norm: str, vol_window: int, eps: float) -> pd.DataFrame:
    """
    Adds:
      r: simple return close/open - 1
      flow: cvd/cv (safe)
      r_star: normalized return
      x: r_star * flow
    """
    df = df.copy()

    df["r"] = df["close"] / df["open"] - 1.0

    # flow in [-1, 1] ideally, but guard divide by zero
    df["flow"] = df["cvd"] / (df["cv"].abs() + eps)
    df["flow"] = df["flow"].clip(-1.0, 1.0)

    if r_norm == "range":
        # near [-1,1] style: where close lands inside range
        rng = (df["high"] - df["low"]).abs() + eps
        df["r_star"] = (df["close"] - df["open"]) / rng
        # optional clip for stability
        df["r_star"] = df["r_star"].clip(-1.5, 1.5)

    elif r_norm == "vol_tanh":
        # normalize by rolling std of returns, then squash
        sigma = df["r"].rolling(vol_window, min_periods=max(10, vol_window // 5)).std()
        z = df["r"] / (sigma + eps)
        df["r_star"] = np.tanh(z)
    else:
        raise ValueError(f"Unknown r_norm={r_norm}")

    df["x"] = df["r_star"] * df["flow"]
    return df


# -----------------------
# Impulse detection
# -----------------------
def _quantile_thresholds(df: pd.DataFrame, q_vol: float, q_flow: float, q_align: float, q_low_vol: float, q_low_flow: float) -> Dict[str, float]:
    abs_r = df["r_star"].abs()
    abs_flow = df["flow"].abs()
    x = df["x"]

    th = {
        "abs_r_hi": float(abs_r.quantile(q_vol)),
        "abs_flow_hi": float(abs_flow.quantile(q_flow)),
        "x_hi": float(x.quantile(q_align)),
        "abs_r_lo": float(abs_r.quantile(q_low_vol)),
        "abs_flow_lo": float(abs_flow.quantile(q_low_flow)),
    }
    return th


def detect_impulses(df: pd.DataFrame, thresholds: Dict[str, float], min_impulse_bars: int) -> pd.DataFrame:
    """
    Returns events df with:
      impulse_id, start_ts, end_ts, direction, n_bars,
      start_idx, end_idx
    """
    abs_r = df["r_star"].abs().values
    abs_flow = df["flow"].abs().values
    x = df["x"].values

    start_cond = (abs_r > thresholds["abs_r_hi"]) & (abs_flow > thresholds["abs_flow_hi"]) & (x > thresholds["x_hi"])
    # end cond: any of these
    end_cond = (x < 0.0) | (abs_r < thresholds["abs_r_lo"]) | (abs_flow < thresholds["abs_flow_lo"])

    events = []
    in_event = False
    s = None
    impulse_id = 0

    for i in range(len(df)):
        if not in_event:
            if start_cond[i]:
                in_event = True
                s = i
        else:
            if end_cond[i]:
                e = i
                n = e - s + 1
                if n >= min_impulse_bars:
                    impulse_id += 1
                    start_ts = df.loc[s, "ts"]
                    end_ts = df.loc[e, "ts"]
                    direction = "UP" if df.loc[e, "close"] >= df.loc[s, "open"] else "DOWN"
                    events.append({
                        "impulse_id": impulse_id,
                        "start_idx": int(s),
                        "end_idx": int(e),
                        "start_ts": start_ts,
                        "end_ts": end_ts,
                        "direction": direction,
                        "n_bars": int(n),
                    })
                in_event = False
                s = None

    # if it ends in an event, close it at last bar
    if in_event and s is not None:
        e = len(df) - 1
        n = e - s + 1
        if n >= min_impulse_bars:
            impulse_id += 1
            start_ts = df.loc[s, "ts"]
            end_ts = df.loc[e, "ts"]
            direction = "UP" if df.loc[e, "close"] >= df.loc[s, "open"] else "DOWN"
            events.append({
                "impulse_id": impulse_id,
                "start_idx": int(s),
                "end_idx": int(e),
                "start_ts": start_ts,
                "end_ts": end_ts,
                "direction": direction,
                "n_bars": int(n),
            })

    return pd.DataFrame(events)


# -----------------------
# IO
# -----------------------
def read_1m_bars(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # normalize ts column name
    if "ts" not in df.columns:
        # if previous code used "ts" after reset_index() it's there.
        # but just in case:
        if "index" in df.columns:
            df = df.rename(columns={"index": "ts"})
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def write_parquet(df: pd.DataFrame, path: Path, compression: str = "zstd") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, tmp, compression=compression)
    tmp.replace(path)


def run(args: Args) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)

    bars_files = sorted(args.bars_1m_dir.glob("*.parquet"))
    if not bars_files:
        raise SystemExit("No 1m bar parquet files found")

    features_dir = args.out_dir / "features"
    impulses_dir = args.out_dir / "impulses"

    for bp in bars_files:
        logger.info("PROCESS %s", bp.name)

        # output naming
        stem = bp.stem  # e.g. bars-1m_20251216 or similar
        feat_path = features_dir / f"{stem.replace('bars-1m', f'feature-{args.tf}')}.parquet"
        imp_path = impulses_dir / f"{stem.replace('bars-1m', f'impulses-{args.tf}')}.parquet"

        if parquet_exists(feat_path) and parquet_exists(imp_path) and not args.overwrite:
            logger.info("SKIP (exists) %s", stem)
            continue

        df_1m = read_1m_bars(bp)
        df_tf = resample_1m_to_tf(df_1m, args.tf)
        df_tf = compute_features(df_tf, args.r_norm, args.vol_window, args.eps)

        th = _quantile_thresholds(df_tf, args.q_vol, args.q_flow, args.q_align, args.q_low_vol, args.q_low_flow)
        events = detect_impulses(df_tf, th, args.min_impulse_bars)

        # store thresholds used (helpful for audit)
        for k, v in th.items():
            df_tf[f"th_{k}"] = v

        write_parquet(df_tf, feat_path)
        write_parquet(events, imp_path)

        logger.info(
            "OK %s: %d TF bars, %d impulses | thresholds=%s",
            stem, len(df_tf), len(events), {k: round(v, 6) for k, v in th.items()}
        )


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.log_level)
    run(args)
