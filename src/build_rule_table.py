import itertools
import pandas as pd
import numpy as np
from pathlib import Path

# display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)

EPS = 1e-12
DATA_PATH = r"D:\data\profile-regime\experiments\binance\futures_um\symbol=BTCUSDT\event_features\event_horizon_features.parquet"

USE_BAND_FILTER = True
BAND_FILTER = {
    "tf": ["15min"],
    "w": [96],
    "m": [2, 4],
    "g": [0, 1],
    "h": [16],
}

FEATURES = ["cv", "cvd", "ncvd", "ret_over_ncvd", "event_ret", "event_range", "event_volatility"]
FEATURE_PAIRS = list(itertools.combinations(FEATURES, 2))

TP_LIST = [0.01, 0.015, 0.02]
SL_LIST = [0.005, 0.007, 0.01]

hz_features = ["hz_ret", "hz_max_up", "hz_max_dn"]

QX = 5
QY = 5

BOTH_HIT_POLICY = "sl_first"

TRAIN_FRAC = 0.8
TRAIN_MIN_COUNT_PER_CELL = 100
TEST_MIN_COUNT_PER_CELL = TRAIN_MIN_COUNT_PER_CELL * (1 - TRAIN_FRAC)

OUT_DIR = Path(r"D:\data\profile-regime\experiments\binance\futures_um\symbol=BTCUSDT\event_features\rules")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def apply_band_filter(df: pd.DataFrame) -> pd.DataFrame:
    if not USE_BAND_FILTER:
        return df
    mask = np.ones(len(df), dtype=bool)
    for k, v in BAND_FILTER.items():
        if k not in df.columns:
            raise ValueError(f"Band filter column missing: {k}")
        mask &= df[k].isin(v)
    return df.loc[mask].copy()

def compute_trade_return(df: pd.DataFrame, tp: float, sl: float) -> pd.DataFrame:
    up_hit = df["hz_max_up"] >= tp
    down_hit = df["hz_max_dn"] <= -sl
    both_hit = up_hit & down_hit

    r = df["hz_ret"].astype(float).copy()

    r = r.where(~up_hit, tp)
    r = r.where(~down_hit, -sl)

    if BOTH_HIT_POLICY == "sl_first":
        r = r.where(~both_hit, -sl)
    elif BOTH_HIT_POLICY == "tp_first":
        r = r.where(~both_hit, tp)
    elif BOTH_HIT_POLICY == "drop":
        r = r.where(~both_hit, np.nan)
    else:
        raise ValueError(f"Unknown BOTH_HIT_POLICY: {BOTH_HIT_POLICY}")

    return r

def make_quantile_edges(x: pd.Series, q: int) -> pd.DataFrame:
    x = x.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(x) == 0:
        return None
    probs = np.linspace(0, 1, q + 1)
    edges = np.quantile(x.to_numpy(), probs)
    edges = np.unique(edges)
    if len(edges) < 3:
        return None
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges

def summarize_by_bins(df: pd.DataFrame, xcol: str, ycol: str, x_edges: np.ndarray, y_edges: np.ndarray, tp: float, sl: float, MIN_COUNT_PER_CELL: int) -> pd.DataFrame:
    need = [xcol, ycol] + hz_features
    missing = [n for n in need if not n in df.columns]
    if missing:
        raise ValueError(f"{missing} not in event dataframe")

    d = df.replace([np.inf, -np.inf], np.nan).dropna(subset=need).copy()

    d["trade_ret"] = compute_trade_return(d, tp=tp, sl=sl)
    d = d.dropna(subset=["trade_ret"]) # when BOTH_HIT_POLICY == "drop"

    d["only_tp_rate"] = (d["hz_max_up"] >= tp) & (d["hz_max_dn"] > -sl)
    d["only_sl_rate"] = (d["hz_max_up"] < tp) & (d["hz_max_dn"] <= -sl)
    d["both_rate"] = (d["hz_max_up"] >= tp) & (d["hz_max_dn"] <= -sl)
    d["win_rate"] = d["trade_ret"] > 0

    d["bin_x"] = pd.cut(d[xcol], bins=x_edges, include_lowest=True)
    d["bin_y"] = pd.cut(d[ycol], bins=y_edges, include_lowest=True)

    g = d.groupby(["bin_x", "bin_y"], observed=False)
    out = g.agg(
        count=("trade_ret", "size"),
        mean_ret=("trade_ret", "mean"),
        median_ret=("trade_ret", "median"),
        std_ret=("trade_ret", "std"),
        only_tp_rate=("only_tp_rate", "mean"),
        only_sl_rate=("only_sl_rate", "mean"),
        both_rate=("both_rate", "mean"),
        win_rate=("win_rate", "mean"),
    ).reset_index()

    # score / min count filter
    out["std_ret"] = out["std_ret"].fillna(0.0)
    out["score"] = (out["mean_ret"] * np.sqrt(out["count"])) / (out["std_ret"] + EPS)
    out.loc[out["count"] < MIN_COUNT_PER_CELL, "score"] = np.nan

    return out

def main():
    df = pd.read_parquet(DATA_PATH)
    df = apply_band_filter(df)

    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["entry_ts"]).sort_values("entry_ts").reset_index(drop=True)

    split_idx = int(len(df) * TRAIN_FRAC)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    print(f"Row after band filter: {len(df)} | train={len(train_df)} | test={len(test_df)}")

    all_rows = []

    for (xcol, ycol) in FEATURE_PAIRS:
        x_edges = make_quantile_edges(train_df[xcol], QX)
        y_edges = make_quantile_edges(train_df[ycol], QY)
        if (x_edges is None) or (y_edges is None):
            continue

        for (tp, sl) in itertools.product(TP_LIST, SL_LIST):
            try:
                train_tab = summarize_by_bins(train_df, xcol, ycol, x_edges, y_edges, tp, sl, TRAIN_MIN_COUNT_PER_CELL)
                test_tab = summarize_by_bins(test_df, xcol, ycol, x_edges, y_edges, tp, sl, TEST_MIN_COUNT_PER_CELL)
            except Exception as e:
                print(f"SKIP pair=({xcol}, {ycol}) tp={tp} sl={sl} -> {e}")
                continue

            train_candidates = train_tab.dropna(subset=["score"]).copy()
            if train_candidates.empty:
                continue

            for r in train_candidates.itertuples(index=False):
                mask = (test_tab["bin_x"] == r.bin_x) & (test_tab["bin_y"] == r.bin_y)
                if mask.any():
                    t = test_tab.loc[mask].iloc[0]
                    test_count = int(t["count"])
                    test_mean = float(t["mean_ret"])
                    test_std = float(t["std_ret"])
                    test_win = float(t["win_rate"])
                    test_score = float(t["score"]) if pd.notna(t["score"]) else np.nan
                else:
                    test_count, test_mean, test_std, test_win, test_score = 0, np.nan, np.nan, np.nan, np.nan

                all_rows.append({
                    "xcol": xcol,
                    "ycol": ycol,
                    "tp": tp,
                    "sl": sl,
                    "bin_x": r.bin_x,
                    "bin_y": r.bin_y,

                    "train_count": int(r.count),
                    "train_mean_ret": float(r.mean_ret),
                    "train_std_ret": float(r.std_ret),
                    "train_win_rate": float(r.win_rate),
                    "train_only_tp_rate": float(r.only_tp_rate),
                    "train_only_sl_rate": float(r.only_sl_rate),
                    "train_both_rate": float(r.both_rate),
                    "train_score": float(r.score) if pd.notna(r.score) else np.nan,

                    "test_count": test_count,
                    "test_mean_ret": test_mean,
                    "test_std_ret": test_std,
                    "test_win_rate": test_win,
                    "test_score": test_score,
                })

    if not all_rows:
        raise RuntimeError(f"No rule tables were generated. Check columns / filters / data.")

    rule_df = pd.DataFrame(all_rows)
    rule_df = rule_df.sort_values(
        ["train_score", "test_mean_ret", "test_count"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    # print
    print("\n=== TOP 20 RULES (train-ranked, with test check) ===")
    cols = [
        "xcol", "ycol", "tp", "sl", "bin_x", "bin_y",
        "train_count", "train_mean_ret", "train_std_ret", "train_win_rate", "train_score",
        "test_count", "test_mean_ret", "test_std_ret", "test_win_rate", "test_score"
    ]
    print(rule_df[cols].head(20))

    # save
    out_parquet = OUT_DIR / "rule_table_train_test.parquet"
    out_csv = OUT_DIR / "rule_table_train_test.csv"
    rule_df.to_parquet(out_parquet, index=False)
    rule_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nSaved -> {out_parquet}")
    print(f"Saved -> {out_csv}")

if __name__ == "__main__":
    main()