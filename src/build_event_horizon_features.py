import numpy as np
import pandas as pd
from pathlib import Path

EPS = 1e-12

SYMBOL = "BTCUSDT"
TF = "15min"

FEATURE_TF_PATH = Path(
    rf"D:\data\profile-regime\cache\binance\futures_um\symbol=BTCUSDT\features_tf\tf={TF}"
)

EVENT_ROOT = Path(
    r"D:\data\profile-regime\experiments\binance\futures_um\symbol=BTCUSDT"
)

OUT_DIR = EVENT_ROOT / "event_features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_grid_meta(grid_name: str) -> dict:
    # grid=tf=15min__w=96__q=0.90__m=2__g=0__h=16
    out= {}
    for part in grid_name.replace("grid=", "").split("__"):
        k, v = part.split("=")
        out[k] = v

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

def main():
    tf_df = pd.read_parquet(FEATURE_TF_PATH).reset_index(drop=True)

    # ts -> datetime
    tf_df["ts"] = pd.to_datetime(tf_df["ts"], utc=True, errors="coerce")
    tf_df = tf_df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    tf_delta = tf_to_timedelta(TF)

    event_paths = list(EVENT_ROOT.rglob("grid=*/events/atz/*.parquet"))
    print(f"Found {len(event_paths)} event parquet files")

    rows = []
    skipped_tf = 0
    skipped_bad_ts = 0
    skipped_no_hz = 0
    skipped_schema = 0

    ts_values = tf_df["ts"].to_numpy()

    for event_path in event_paths:
        try:
            ev_df = pd.read_parquet(event_path)
        except Exception:
            continue

        if ("start_ts" not in ev_df.columns) or ("end_ts" not in ev_df.columns):
            skipped_schema += 1
            continue

        grid = event_path.parts[-4]
        meta = parse_grid_meta(grid)

        if meta["tf"] != TF:
            skipped_tf += 1
            continue

        h = meta["h"]

        ev_df["start_ts"] = pd.to_datetime(ev_df["start_ts"], utc=True, errors="coerce")
        ev_df["end_ts"] = pd.to_datetime(ev_df["end_ts"], utc=True, errors="coerce")
        ev_df = ev_df.dropna(subset=["start_ts", "end_ts"])
        if ev_df.empty:
            continue

        for ev in ev_df.itertuples(index=False):
            start_ts = ev.start_ts
            end_ts = ev.end_ts

            mask_ev = (tf_df["ts"] >= start_ts) & (tf_df["ts"] <= end_ts)
            ev_seg = tf_df.loc[mask_ev]

            if ev_seg.empty:
                skipped_bad_ts += 1
                continue

            entry_ts = end_ts + tf_delta
            hz_start = int(np.searchsorted(ts_values, entry_ts, side="right"))
            hz_end = hz_start + h

            if hz_start >= len(tf_df):
                skipped_no_hz += 1
                continue

            if hz_end > len(tf_df):
                skipped_no_hz += 1
                continue

            hz_seg = tf_df.iloc[hz_start:hz_end]
            if hz_seg.empty:
                skipped_no_hz += 1
                continue

            # event features
            cv = float(ev_seg["cv"].sum())
            cvd = float(ev_seg["cvd"].sum())
            ncvd = cvd / (cv + EPS)

            ev_open = float(ev_seg["open"].iloc[0])
            ev_close = float(ev_seg["close"].iloc[-1])
            event_ret = ev_close / ev_open - 1.0

            ret_over_ncvd = event_ret / (ncvd + EPS)

            # horizon features
            hz_open = float(hz_seg["open"].iloc[0])
            hz_close = float(hz_seg["close"].iloc[-1])

            hz_ret = hz_close / hz_open - 1.0
            hz_max_up = float(hz_seg["high"].max() / hz_open - 1.0)
            hz_max_dn = float(hz_seg["low"].min() / hz_open - 1.0)
            hz_vol = float(hz_seg["close"].pct_change().std())

            row = {
                # meta
                "symbol": SYMBOL,
                "tf": TF,
                "grid": grid,
                "w": meta["w"],
                "q": meta["q"],
                "m": meta["m"],
                "g": meta["g"],
                "h": meta["h"],
                "atz_id": ev.atz_id,
                "n_bars": ev.n_bars,

                # time
                "start_ts": start_ts,
                "end_ts": end_ts,
                "entry_ts": entry_ts,

                # activity
                "cv": cv,
                "cvd": cvd,
                "ncvd": ncvd,
                "ret_over_ncvd": ret_over_ncvd,
                "trade_count": float(ev_seg["trade_count"].sum()),

                # price (event)
                "event_ret": float(event_ret),
                "event_range": float(ev_seg["high"].max() - ev_seg["low"].min()),
                "event_volatility": float(ev_seg["close"].pct_change().std()),

                # price (horizon)
                "hz_ret": float(hz_ret),
                "hz_max_up": hz_max_up,
                "hz_max_dn": hz_max_dn,
                "hz_vol": float(hz_vol),

                # direction consistency
                "dir_match": int(np.sign(event_ret) == np.sign(hz_ret)),
            }
            rows.append(row)

    out_df = pd.DataFrame(rows)
    print(f"Built {len(out_df)} rows")
    print(f"Skipped tf!={TF} files={skipped_tf}, empty_evseg={skipped_bad_ts}, no_horizon={skipped_no_hz}")

    out_path = OUT_DIR / "event_horizon_features.parquet"
    out_df.to_parquet(out_path, index=False)
    print(f"Saved -> {out_path}")

    # check
    if not out_df.empty:
        print("tf unique:", out_df["tf"].unique())
        print("h unique count:", out_df["h"].nunique())
        print("start_ts < end_ts all:", bool((out_df["start_ts"] < out_df["end_ts"]).all()))
        print("entry_ts >= end_ts all:", bool((out_df["entry_ts"] >= out_df["end_ts"]).all()))

if __name__ == "__main__":
    main()