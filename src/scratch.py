import pandas as pd
import numpy as np
from pathlib import Path

EPS = 1e-12

# =====================
# CONFIG
# =====================
EVENT_PATH = r"D:\data\profile-regime\experiments\binance\futures_um\symbol=BTCUSDT\event_features\event_horizon_features.parquet"
RULE_PATH  = r"D:\data\profile-regime\experiments\binance\futures_um\symbol=BTCUSDT\event_features\rules\rule_table_train_test.csv"

OUT_DIR = Path(r"D:\data\profile-regime\experiments\binance\futures_um\symbol=BTCUSDT\event_features\walkforward")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# rule filter criteria
MIN_TRAIN_SCORE = 2.0
MIN_TEST_SCORE  = 1.0
MIN_TEST_COUNT  = 30

# walk-forward window
TRAIN_DAYS = 90
TEST_DAYS  = 30
STEP_DAYS  = 30

# =====================
# UTIL
# =====================
def compute_trade_return(df, tp, sl):
    up_hit = df["hz_max_up"] >= tp
    dn_hit = df["hz_max_dn"] <= -sl
    both = up_hit & dn_hit

    r = df["hz_ret"].copy()
    r[up_hit] = tp
    r[dn_hit] = -sl
    r[both] = -sl   # conservative

    return r

def max_drawdown(x):
    c = np.cumsum(x)
    peak = np.maximum.accumulate(c)
    return np.min(c - peak)

# =====================
# MAIN
# =====================
def main():
    events = pd.read_parquet(EVENT_PATH)
    events["entry_ts"] = pd.to_datetime(events["end_ts"], utc=True)
    events = events.sort_values("entry_ts").reset_index(drop=True)

    rules = pd.read_csv(RULE_PATH)

    # ---- rule pre-filter ----
    rules = rules[
        (rules["train_score"] >= MIN_TRAIN_SCORE) &
        (rules["test_score"]  >= MIN_TEST_SCORE) &
        (rules["test_count"]  >= MIN_TEST_COUNT)
    ].reset_index(drop=True)

    print(f"Candidate rules: {len(rules)}")

    results = []

    t0 = events["entry_ts"].min()
    t1 = events["entry_ts"].max()

    start = t0

    while start + pd.Timedelta(days=TRAIN_DAYS + TEST_DAYS) <= t1:
        train_end = start + pd.Timedelta(days=TRAIN_DAYS)
        test_end  = train_end + pd.Timedelta(days=TEST_DAYS)

        train_ev = events[(events["entry_ts"] >= start) & (events["entry_ts"] < train_end)]
        test_ev  = events[(events["entry_ts"] >= train_end) & (events["entry_ts"] < test_end)]

        if len(test_ev) < 10:
            start += pd.Timedelta(days=STEP_DAYS)
            continue

        for r in rules.itertuples(index=False):
            mask = (
                (train_ev[r.xcol].between(r.bin_x.left, r.bin_x.right)) &
                (train_ev[r.ycol].between(r.bin_y.left, r.bin_y.right))
            )
            if mask.sum() < 20:
                continue

            test_mask = (
                (test_ev[r.xcol].between(r.bin_x.left, r.bin_x.right)) &
                (test_ev[r.ycol].between(r.bin_y.left, r.bin_y.right))
            )

            d = test_ev.loc[test_mask].copy()
            if len(d) < 10:
                continue

            d["ret"] = compute_trade_return(d, r.tp, r.sl)

            results.append({
                "xcol": r.xcol,
                "ycol": r.ycol,
                "tp": r.tp,
                "sl": r.sl,
                "bin_x": r.bin_x,
                "bin_y": r.bin_y,
                "start": start,
                "mean_ret": d["ret"].mean(),
                "win_rate": (d["ret"] > 0).mean(),
                "count": len(d),
                "sharpe_like": d["ret"].mean() / (d["ret"].std() + EPS),
                "max_dd": max_drawdown(d["ret"].values),
            })

        start += pd.Timedelta(days=STEP_DAYS)

    res = pd.DataFrame(results)
    res = res.sort_values("sharpe_like", ascending=False)

    out = OUT_DIR / "walkforward_results.csv"
    res.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"Saved -> {out}")
    print(res.head(10))

if __name__ == "__main__":
    main()
