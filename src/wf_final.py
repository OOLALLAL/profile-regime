import itertools
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import numpy as np


# display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)

# =========================
# CONFIG
# =========================
DATA_PATH = r"D:\data\profile-regime\experiments\binance\futures_um\symbol=BTCUSDT\event_features\event_horizon_features.parquet"
OUT_DIR = Path(r"D:\data\profile-regime\experiments\binance\futures_um\symbol=BTCUSDT\event_features\wf_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
HZ_FEATURES = ["hz_ret", "hz_high_path", "hz_low_path"]

QX, QY = 7, 7
TP_LIST = [0.012]
SL_LIST = [0.012]

BOTH_HIT_POLICY = "sl_first" # "sl_first" | "tp_first" | "drop"
COMMISSION = 0.0004
SLIPPAGE = 0.0001

MAX_CONCURRENT = None
# None => no cap (pure add)
# int  => cap max concurrent entries per row (pick strongest first)

# Selection filters
TOP_K_RULES = 30
TRAIN_MIN_COUNT_PER_CELL = 100
VAL_MIN_COUNT_PER_CELL = 50
MAX_BOTH_RATE = 0.05
VAL_MEAN_RET = 0

HOLDOUT_FRAC = 0.20
WINDOW_NUM = 5
TRAIN_FRAC_IN_WINDOW = 0.70
MIN_ROWS_PER_WINDOW = 100

EPS = 1e-12

# =========================
# Data structures
# =========================
@dataclass(frozen=True)
class Rule:
    side: str
    xcol: str
    ycol: str
    tp: float
    sl: float
    qx_id: int
    qy_id: int
    strength: float # train_mean_ret (or train_score) used ONLY for tie-breaking / cap ordering

# =========================
# Utils
# =========================
def apply_band_filter(df: pd.DataFrame) -> pd.DataFrame:
    if not USE_BAND_FILTER:
        return df
    mask = np.ones(len(df), dtype=bool)
    for k, v in BAND_FILTER.items():
        if k not in df.columns:
            raise ValueError(f"Band filter column missing: {k}")
        mask &= df[k].isin(v)
    return df.loc[mask].copy()

def make_quantile_edges(x: pd.Series, q: int) -> pd.DataFrame | None:
    x = x.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(x) < q:
        return None
    probs = np.linspace(0, 1, q + 1)
    edges = np.quantile(x.to_numpy(), probs)
    edges = np.unique(edges.astype(float))
    if len(edges) < 3:
        return None
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges

def bin_id_from_edges(values: pd.Series, edges: np.ndarray) -> pd.Series:
    v = values.astype(float).to_numpy()
    idx = np.searchsorted(edges, v, side="right") - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    return pd.Series(idx, index=values.index, dtype="int16")

def _first_idx_ge(path_list: list[float], thr: float) -> int | None:
    for i, v in enumerate(path_list):
        if v >= thr:
            return i
    return None

def _first_idx_le(path_list: list[float], thr: float) -> int | None:
    for i, v in enumerate(path_list):
        if v <= thr:
            return i
    return None

def compute_trade_return(df: pd.DataFrame, tp: float, sl: float, side: str) -> pd.DataFrame:
    hz_ret = df["hz_ret"].astype(float).to_numpy()
    high_paths = df["hz_high_path"].to_list()
    low_paths = df["hz_low_path"].to_list()

    out = np.full(len(df), np.nan, dtype="float64")
    for i in range(len(df)):
        hp = high_paths[i]
        lp = low_paths[i]

        if not isinstance(hp, (list, tuple)) or not isinstance(lp, (list, tuple)) or len(hp) == 0 or len(lp) == 0:
            out[i] = hz_ret[i] if side == "long" else -hz_ret[i]
            continue

        if side == "long":
            tp_i = _first_idx_ge(hp, tp)
            sl_i = _first_idx_le(lp, -sl)
            nohit_ret = hz_ret[i]
            tp_ret, sl_ret = tp, -sl

        elif side == "short":
            tp_i = _first_idx_le(lp, -tp)
            sl_i = _first_idx_ge(hp, sl)
            nohit_ret = -hz_ret[i]
            tp_ret, sl_ret = tp, -sl

        else:
            raise ValueError(f"bad side={side}")

        if tp_i is None and sl_i is None:
            out[i] = nohit_ret
        elif tp_i is not None and sl_i is None:
            out[i] = tp_ret
        elif tp_i is None and sl_i is not None:
            out[i] = sl_ret
        else:
            if tp_i < sl_i:
                out[i] = tp_ret
            elif sl_i < tp_i:
                out[i] = sl_ret
            else:
                if BOTH_HIT_POLICY == "sl_first":
                    out[i] = sl_ret
                elif BOTH_HIT_POLICY == "tp_first":
                    out[i] = tp_ret
                elif BOTH_HIT_POLICY == "drop":
                    out[i] = np.nan
                else:
                    raise ValueError(f"Unknown BOTH_HIT_POLICY: {BOTH_HIT_POLICY}")

    r = pd.Series(out, index=df.index)
    return r - (COMMISSION + SLIPPAGE)

def summarize_cells(df: pd.DataFrame, xcol: str, ycol: str, x_edges: np.ndarray, y_edges: np.ndarray, tp: float, sl: float, side: str) -> pd.DataFrame:
    need = [xcol, ycol] + HZ_FEATURES
    d = df.copy()

    num_cols = [xcol, ycol, "hz_ret"]
    for c in num_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
        d[c] = d[c].replace([np.inf, -np.inf], np.nan)

    d = d.dropna(subset=num_cols + ["hz_high_path", "hz_low_path"]).copy()
    if d.empty:
        return pd.DataFrame()

    d["trade_ret"] = compute_trade_return(d, tp, sl, side=side)
    d = d.dropna(subset=["trade_ret"]) # when BOTH_HIT_POLICY == "drop"

    d["qx_id"] = bin_id_from_edges(d[xcol], x_edges) # 0 ~ (QX-1)
    d["qy_id"] = bin_id_from_edges(d[ycol], y_edges) # 0 ~ (QY-1)

    def _both_hit_row(hp, lp):
        if not isinstance(hp, (list, tuple)) or not isinstance(lp, (list, tuple)):
            return False
        return (_first_idx_ge(hp, tp) is not None) and (_first_idx_le(lp, -sl) is not None)
    d["both"] = [
        _both_hit_row(hp, lp) for hp, lp in zip(d["hz_high_path"].to_list(), d["hz_low_path"].to_list())
    ]

    d["win"] = d["trade_ret"] > 0

    g = d.groupby(["qx_id", "qy_id"], observed=True)
    out = g.agg(
        count=("trade_ret", "size"),
        mean_ret=("trade_ret", "mean"),
        std_ret=("trade_ret", "std"),
        win_rate=("win", "mean"),
        both_rate=("both", "mean"),
    ).reset_index()

    # score / min count filter
    out["std_ret"] = out["std_ret"].fillna(0.0)
    out["score"] = (out["mean_ret"] * np.sqrt(out["count"])) / (out["std_ret"] + EPS)
    return out

def build_rule_table(train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side in ["long", "short"]:
        for (xcol, ycol) in FEATURE_PAIRS:
            x_edges = make_quantile_edges(train_df[xcol], QX)
            y_edges = make_quantile_edges(train_df[ycol], QY)
            if (x_edges is None) or (y_edges is None):
                continue

            for tp, sl in itertools.product(TP_LIST, SL_LIST):
                tr = summarize_cells(train_df, xcol, ycol, x_edges, y_edges, tp, sl, side)
                if tr.empty:
                    continue
                va = summarize_cells(val_df, xcol, ycol, x_edges, y_edges, tp, sl, side)

                va_map = {(int(r.qx_id), int(r.qy_id)): r for r in va.itertuples(index=False)} if not va.empty else {}

                for r in tr.itertuples(index=False):
                    v = va_map.get((int(r.qx_id), int(r.qy_id)))
                    rows.append({
                        "side": side,
                        "xcol": xcol, "ycol": ycol, "tp": float(tp), "sl": float(sl),
                        "qx_id": int(r.qx_id), "qy_id": int(r.qy_id),

                        "train_count": int(r.count),
                        "train_mean_ret": float(r.mean_ret),
                        "train_score": float(r.score) if pd.notna(r.score) else np.nan,
                        "train_both_rate": float(r.both_rate),

                        "val_count": int(v.count) if v is not None else 0,
                        "val_mean_ret": float(v.mean_ret) if v is not None else np.nan,
                        "val_score": float(v.score) if (v is not None and pd.notna(v.score)) else np.nan,
                    })
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def select_rules(rule_table: pd.DataFrame) -> list[Rule]:
    rt = rule_table.copy()
    rt = rt[rt["train_count"] >= TRAIN_MIN_COUNT_PER_CELL]
    rt = rt[rt["val_count"] >= VAL_MIN_COUNT_PER_CELL]
    rt = rt[rt["train_both_rate"] <= MAX_BOTH_RATE]
    rt = rt[rt["val_mean_ret"].notna()]
    rt = rt[rt["val_mean_ret"] > VAL_MEAN_RET]

    if rt.empty:
        return []

    rt = rt.sort_values(
        ["val_mean_ret", "val_score", "val_count", "train_count"],
        ascending=[False, False, False, False],
        na_position="last",
    ).head(TOP_K_RULES)

    # strength for cap ordering (train-based only; no leakage into test)
    # choose one:
    #   strength = train_mean_ret  (simple)
    #   strength = train_score     (risk-adjusted-ish)
    strength = rt["train_mean_ret"].fillna(-np.inf).to_numpy()

    out = []
    for i, r in enumerate(rt.itertuples(index=False)):
        out.append(Rule(
            r.side,
            r.xcol, r.ycol, float(r.tp), float(r.sl), int(r.qx_id), int(r.qy_id),
            float(strength[i]),
        ))
    out.sort(key=lambda z: z.strength, reverse=True)
    return out

def apply_rules_add_capped(df: pd.DataFrame, rules: list[Rule], edges_by_col: dict[str, np.ndarray]) -> tuple[pd.Series, pd.Series]:
    if not rules or df.empty:
        return pd.Series(0.0, index=df.index), pd.Series(0, index=df.index, dtype="int16")

    used_cols = sorted({r.xcol for r in rules} | {r.ycol for r in rules})
    bins = {}
    for col in used_cols:
        e = edges_by_col.get(col)
        if e is None:
            return pd.Series(0.0, index=df.index), pd.Series(0, index=df.index, dtype="int16")
        bins[col] = bin_id_from_edges(df[col], e)

    tr_by_key: dict[tuple[float, float], pd.Series] = {}
    for r in rules:
        tr_by_key.setdefault((r.side, r.tp, r.sl), None)
    for (side, tp, sl) in list(tr_by_key.keys()):
        tr_by_key[(side, tp, sl)] = compute_trade_return(df, tp, sl, side=side).fillna(0.0)

    pnl = pd.Series(0.0, index=df.index)
    trig_count = pd.Series(0, index=df.index, dtype="int16")

    if MAX_CONCURRENT is None:
        for r in rules:
            mask = (bins[r.xcol] == r.qx_id) & (bins[r.ycol] == r.qy_id)
            if mask.any():
                trig_count.loc[mask] += 1
                pnl.loc[mask] += tr_by_key[(r.side, r.tp, r.sl)].loc[mask]
        return pnl, trig_count

    filled = pd.Series(0, index=df.index, dtype="int16")
    for r in rules:
        mask = (bins[r.xcol] == r.qx_id) & (bins[r.ycol] == r.qy_id)
        if not mask.any():
            continue

        trig_count.loc[mask] += 1

        mask2 = mask & (filled < MAX_CONCURRENT)
        if mask2.any():
            pnl.loc[mask2] += tr_by_key[(r.side, r.tp, r.sl)].loc[mask2]
            filled.loc[mask2] += 1

    return pnl, trig_count

def summarize_pnl(pnl: pd.Series) -> dict:
    pnl = pnl.astype(float)
    eq = pnl.cumsum()
    peak = eq.cummax()
    dd = eq - peak
    return {
        "mean_ret": float(pnl.mean()),
        "sum_ret": float(pnl.sum()),
        "trade_rate": float((pnl != 0).mean()),
        "win_rate": float((pnl > 0).mean()),
        "mdd": float(dd.min()),
    }

def summarize_triggers(k: pd.Series) -> dict:
    k = k.astype(int)
    return {
        "k_mean": float(k.mean()),
        "k_max": int(k.max()),
        "k_p95": int(np.quantile(k.to_numpy(), 0.95)),
        "k_nonzero_rate": float((k > 0).mean()),
    }

def make_time_windows(df: pd.DataFrame, n_windows: int) -> list[tuple[int, int]]:
    tvals = df["entry_ts"].astype("int64").to_numpy()
    borders = np.quantile(tvals, np.linspace(0, 1, n_windows + 1))
    borders[0] = tvals.min()
    borders[-1] = tvals.max() + 1

    ranges = []
    for i in range(n_windows):
        mask = (tvals >= borders[i]) & (tvals < borders[i + 1])
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        ranges.append((int(idx[0]), int(idx[-1] + 1)))
    return ranges

# =========================
# Main
# =========================
def main():
    df = pd.read_parquet(DATA_PATH)
    df = apply_band_filter(df)

    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["entry_ts"]).sort_values("entry_ts").reset_index(drop=True)
    print(f"Rows(event + horizon) after filter: {len(df)}")

    must_cols = ["hz_high_path", "hz_low_path"]
    miss = [c for c in must_cols if c not in df.columns]
    if miss:
        raise ValueError(
            f"Missing columns in DATA_PATH: {miss}. Rebuild event_horizon_features.parquet with path columns.")

    n = len(df)
    cut = int(n * (1 - HOLDOUT_FRAC))
    dev = df.iloc[:cut].copy()
    holdout = df.iloc[cut:].copy()
    print(f"Development set={len(dev)} // Final test set={len(holdout)}")
    print(f"MAX_CONCURRENT (position)={MAX_CONCURRENT}")

    dev_ranges = make_time_windows(dev, WINDOW_NUM)
    if len(dev_ranges) < 3:
        raise RuntimeError("Not enough windows in dev. Reduce WINDOW_NUM or add data.")

    wf_rows = []
    deployed_rules: list[Rule] = []
    deployed_edges: dict[str, np.ndarray] = {}

    for i in range(len(dev_ranges) - 1):
        l0, r0 = dev_ranges[i]
        l1, r1 = dev_ranges[i + 1]
        block = df.iloc[l0:r0].copy()
        test_block = df.iloc[l1:r1].copy()

        if len(block) < MIN_ROWS_PER_WINDOW or len(test_block) < MIN_ROWS_PER_WINDOW:
            continue

        split = int(len(block) * TRAIN_FRAC_IN_WINDOW)
        train_block = block[:split].copy()
        val_block = block[split:].copy()

        rule_table = build_rule_table(train_block, val_block)
        rules = select_rules(rule_table)

        edges_by_col = {}
        ok = True
        for col in sorted({r.xcol for r in rules} | {r.ycol for r in rules}):
            e = make_quantile_edges(train_block[col], QX) # (QX==QY)
            if e is None:
                ok = False
                break
            edges_by_col[col] = e
        if not ok:
            rules = []
            edges_by_col = {}

        deployed_rules = rules
        deployed_edges = edges_by_col

        pnl, k = apply_rules_add_capped(test_block, deployed_rules, deployed_edges)
        summ = summarize_pnl(pnl)
        ks = summarize_triggers(k)

        wf_rows.append({
            "wf_idx": i,
            "train_rows": len(train_block),
            "val_rows": len(val_block),
            "test_rows": len(test_block),
            "n_rules": len(deployed_rules),
            **summ,
            **ks,
        })

        if not rule_table.empty:
            rule_table.sort_values(["val_mean_ret", "val_score", "val_count"], ascending=False) \
                .head(50) \
                .to_csv(OUT_DIR / f"wf_{i:02d}_rule_table_top50.csv", index=False, encoding="utf-8-sig")

    wf_df = pd.DataFrame(wf_rows)
    wf_df.to_csv(OUT_DIR / "walk_forward_test_summary.csv", index=False, encoding="utf-8-sig")
    print("\n=== WALK-FORWARD (TEST windows stitched) ===")
    print(wf_df.to_string(index=False))

    if wf_df.empty:
        print("\nFINAL: FAIL (no usable WF windows)")
        return

    stitched_sum = float(wf_df["sum_ret"].sum())
    stitched_rows = int(wf_df["test_rows"].sum())
    stitched_mean = stitched_sum / max(1, stitched_rows)
    print(f"\nDEV stitched sum_ret={stitched_sum:.6g}  approx_mean_per_row={stitched_mean:.6g}")

    # ===== HOLDOUT (ONE-SHOT) =====
    if not deployed_rules:
        print("\nHOLDOUT: FAIL (no deployed rules)")
        return

    pnl_h, k_h = apply_rules_add_capped(holdout, deployed_rules, deployed_edges)
    h = summarize_pnl(pnl_h)
    hk = summarize_triggers(k_h)

    print("\n=== HOLDOUT (ONE-SHOT) ===")
    for k, v in h.items():
        print(f"{k}: {v}")
    print("trigger_stats:", hk)

    decision = "PASS" if (h["mean_ret"] > 0 and h["sum_ret"] > 0) else "FAIL"
    print(f"\nFINAL DECISION: {decision}")

    pd.DataFrame([{"decision": decision, **h, **hk}]).to_csv(OUT_DIR / "holdout_result.csv", index=False,
                                                             encoding="utf-8-sig")

if __name__ == "__main__":
    main()