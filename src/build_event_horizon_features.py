import numpy as np
import pandas as pd
from pathlib import Path

EPS = 1e-12

SYMBOL = "BTCUSDT"
TF = "15min"

FEATURE_TF_PATH = Path(
    rf"D:\data\profile-regime\cache\binance\futures_um\symbol=BTCUSDT\features_tf\tf={TF}"
)

EVENT_ROOT = Path(r"D:\data\profile-regime\experiments\binance\futures_um\symbol=BTCUSDT")
RUN_FILE = "run=20260112_214647_999"

OUT_DIR = EVENT_ROOT / "event_features"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_grid_meta(grid_name: str) -> dict:
    """
    grid=...__tf=15min__w=96__q=0.9__m=2__g=1__h=16 형태를 파싱
    """
    out = {}
    name = grid_name.replace("grid=", "")
    for part in name.split("__"):
        k, v = part.split("=")
        out[k] = v

    # typed
    out["w"] = int(out["w"])
    out["q"] = float(out["q"])
    out["m"] = int(out["m"])
    out["g"] = int(out["g"])
    out["h"] = int(out["h"])
    return out


def tf_to_timedelta(tf: str) -> pd.Timedelta:
    tf = tf.strip().lower()
    if tf.endswith("min"):
        return pd.Timedelta(minutes=int(tf.replace("min", "")))
    if tf.endswith("h"):
        return pd.Timedelta(hours=int(tf.replace("h", "")))
    raise ValueError(f"Unsupported TF format: {tf}")


def safe_read_parquet(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path)
    except Exception as e:
        print(f"[WARN] failed to read {path}: {e}")
        return None


def ensure_sorted_tf(tf_df: pd.DataFrame) -> pd.DataFrame:
    if "ts" not in tf_df.columns:
        raise ValueError("features_tf parquet must contain 'ts' column")

    tf_df = tf_df.copy()
    tf_df["ts"] = pd.to_datetime(tf_df["ts"], utc=True, errors="coerce")
    tf_df = tf_df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return tf_df


def main():
    # --- Load 15m features_tf ---
    tf_df = safe_read_parquet(FEATURE_TF_PATH)
    if tf_df is None or tf_df.empty:
        raise SystemExit(f"Failed to load features_tf: {FEATURE_TF_PATH}")

    tf_df = ensure_sorted_tf(tf_df)

    # sanity columns
    need_cols = ["ts", "open", "high", "low", "close", "cv", "cvd", "trade_count"]
    miss = [c for c in need_cols if c not in tf_df.columns]
    if miss:
        raise ValueError(f"Missing columns in features_tf: {miss}")

    tf_delta = tf_to_timedelta(TF)

    # for fast mapping
    # numpy datetime64[ns] array (timezone-naive in numpy, but consistent)
    ts_values = tf_df["ts"].to_numpy(dtype="datetime64[ns]")

    # --- Scan event parquet files ---
    run_root = EVENT_ROOT / RUN_FILE
    event_paths = sorted(run_root.rglob("grid=*/events/atz/*.parquet"))
    print(f"Found {len(event_paths)} event parquet files")

    rows = []
    skipped_tf = 0
    skipped_bad_ts = 0
    skipped_no_hz = 0
    skipped_schema = 0
    skipped_empty = 0

    for event_path in event_paths:
        ev_df = safe_read_parquet(event_path)
        if ev_df is None or ev_df.empty:
            skipped_empty += 1
            continue

        if ("start_ts" not in ev_df.columns) or ("end_ts" not in ev_df.columns):
            skipped_schema += 1
            continue

        # grid meta
        grid = event_path.parts[-4]
        meta = parse_grid_meta(grid)

        if meta.get("tf") != TF:
            skipped_tf += 1
            continue

        h = meta["h"]

        # normalize event timestamps
        ev_df = ev_df.copy()
        ev_df["start_ts"] = pd.to_datetime(ev_df["start_ts"], utc=True, errors="coerce")
        ev_df["end_ts"] = pd.to_datetime(ev_df["end_ts"], utc=True, errors="coerce")
        ev_df = ev_df.dropna(subset=["start_ts", "end_ts"])
        if ev_df.empty:
            skipped_empty += 1
            continue

        # iterate events
        for ev in ev_df.itertuples(index=False):
            start_ts = ev.start_ts
            end_ts = ev.end_ts

            # event segment on 15m tf_df
            mask_ev = (tf_df["ts"] >= start_ts) & (tf_df["ts"] <= end_ts)
            ev_seg = tf_df.loc[mask_ev]
            if ev_seg.empty:
                skipped_bad_ts += 1
                continue

            # entry at NEXT bar after event end (end_ts + tf_delta)
            entry_ts = (end_ts + tf_delta)

            # IMPORTANT: side="left" to map to exact next bar timestamp if exists
            entry_ts64 = np.datetime64(entry_ts.to_datetime64())
            hz_start = int(np.searchsorted(ts_values, entry_ts64, side="left"))
            hz_end = hz_start + h

            if hz_start < 0 or hz_start >= len(tf_df) or hz_end > len(tf_df):
                skipped_no_hz += 1
                continue

            hz_seg = tf_df.iloc[hz_start:hz_end]
            if hz_seg.empty or len(hz_seg) != h:
                skipped_no_hz += 1
                continue

            # ----- event features -----
            cv = float(ev_seg["cv"].sum())
            cvd = float(ev_seg["cvd"].sum())
            ncvd = float(cvd / (cv + EPS))

            ev_open = float(ev_seg["open"].iloc[0])
            ev_close = float(ev_seg["close"].iloc[-1])
            event_ret = float(ev_close / ev_open - 1.0)

            ret_over_ncvd = float(event_ret / (ncvd + EPS))

            # price (event)
            event_high = float(ev_seg["high"].max())
            event_low = float(ev_seg["low"].min())
            event_range = float(event_high - event_low)
            event_volatility = float(ev_seg["close"].pct_change().std(ddof=0))

            # ----- horizon features -----
            hz_open = float(hz_seg["open"].iloc[0])
            hz_close = float(hz_seg["close"].iloc[-1])

            hz_ret = float(hz_close / hz_open - 1.0)
            hz_max_up = float(hz_seg["high"].max() / hz_open - 1.0)
            hz_max_dn = float(hz_seg["low"].min() / hz_open - 1.0)
            hz_vol = float(hz_seg["close"].pct_change().std(ddof=0))

            # path (store as python list of float to avoid dtype weirdness)
            hz_high_path = (hz_seg["high"].to_numpy(dtype=float) / hz_open - 1.0).tolist()
            hz_low_path = (hz_seg["low"].to_numpy(dtype=float) / hz_open - 1.0).tolist()
            hz_close_path = (hz_seg["close"].to_numpy(dtype=float) / hz_open - 1.0).tolist()

            # mapping diagnostics
            hz_start_ts = tf_df["ts"].iloc[hz_start]
            hz_end_ts = tf_df["ts"].iloc[hz_end - 1]
            entry_gap_sec = float((hz_start_ts - entry_ts).total_seconds())

            # direction consistency
            dir_match = int(np.sign(event_ret) == np.sign(hz_ret))

            row = {
                # meta
                "symbol": SYMBOL,
                "tf": TF,
                "grid": grid,
                "w": meta["w"],
                "q": meta["q"],
                "m": meta["m"],
                "g": meta["g"],
                "h": h,
                "atz_id": getattr(ev, "atz_id", np.nan),
                "n_bars": getattr(ev, "n_bars", np.nan),

                # time
                "start_ts": start_ts,
                "end_ts": end_ts,
                "entry_ts": entry_ts,
                "hz_start_ts": hz_start_ts,
                "hz_end_ts": hz_end_ts,
                "entry_gap_sec": entry_gap_sec,
                "hz_len": int(len(hz_seg)),

                # activity
                "cv": cv,
                "cvd": cvd,
                "ncvd": ncvd,
                "ret_over_ncvd": ret_over_ncvd,
                "trade_count": float(ev_seg["trade_count"].sum()),

                # price (event)
                "event_ret": event_ret,
                "event_range": event_range,
                "event_volatility": event_volatility,

                # price (horizon)
                "hz_ret": hz_ret,
                "hz_max_up": hz_max_up,
                "hz_max_dn": hz_max_dn,
                "hz_vol": hz_vol,

                # paths
                "hz_high_path": hz_high_path,
                "hz_low_path": hz_low_path,
                "hz_close_path": hz_close_path,

                # direction
                "dir_match": dir_match,
            }
            rows.append(row)

    out_df = pd.DataFrame(rows)

    print(f"Built {len(out_df)} rows")
    print(
        f"Skipped tf!={TF} files={skipped_tf}, "
        f"bad_ts={skipped_bad_ts}, "
        f"no_horizon={skipped_no_hz}, "
        f"bad_schema={skipped_schema}, "
        f"empty_or_unreadable={skipped_empty}"
    )

    out_path = OUT_DIR / "event_horizon_features.parquet"
    out_df.to_parquet(out_path, index=False)
    print(f"Saved -> {out_path}")

    # quick checks
    if not out_df.empty:
        print("tf unique:", out_df["tf"].unique())
        print("h unique count:", out_df["h"].nunique())
        print("start_ts < end_ts all:", bool((out_df["start_ts"] < out_df["end_ts"]).all()))
        print("entry_ts >= end_ts all:", bool((out_df["entry_ts"] >= out_df["end_ts"]).all()))

        # Mapping sanity: entry_gap_sec should be 0 (perfect) or small positive
        print("entry_gap_sec summary:")
        print(out_df["entry_gap_sec"].describe(percentiles=[0.0, 0.5, 0.9, 0.99]).to_string())

        # Path sanity: should NOT all be empty
        lens = out_df["hz_high_path"].apply(lambda x: len(x) if isinstance(x, list) else -1)
        print("hz_high_path length stats:", lens.describe().to_string())

        # Optional: if duplicates are possible (same grid, atz_id, entry_ts), you can de-dup:
        # out_df = out_df.drop_duplicates(subset=["grid", "atz_id", "entry_ts"], keep="first")

if __name__ == "__main__":
    main()
