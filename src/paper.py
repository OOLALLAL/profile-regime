# src/scratch3.py
# -*- coding: utf-8 -*-
"""
scratch3.py
- Robust CLI (no argparse conflicts)
- Windows-safe output filenames (short tag + hash)
- Supports:
  --fee-bps, --slippage-bps, --fee-grid
  --nohit-pnl {zero,mtm_last}
  --date-min/--date-max
  --culprit-period/topk/loss-only
  --require_eventret_match
  --dedup-key/keep
  filters: ret_over_ncvd_max, event_vol_max, trade_count_max,
           event_ret_short_max/long_min, ncvd_short_min/long_max
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

EPS = 1e-12


# -----------------------------
# Utils
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_list_maybe(x):
    """Parquet에 list/np.ndarray/문자열('[..]') 등 섞여 있어도 list[float]로 통일."""
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    if isinstance(x, list):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, str):
        s = x.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                v = ast.literal_eval(s)
                if isinstance(v, (list, tuple, np.ndarray)):
                    return list(v)
            except Exception:
                return None
        return None
    return None


def to_utc_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


def to_period_str(ts: pd.Series, split: str) -> pd.Series:
    """
    Period 변환 시 tz warning 피하려면 tz 정보를 제거하고 period로 변환.
    """
    if split is None:
        split = "Q"
    t = pd.to_datetime(ts, utc=True, errors="coerce")
    # Period는 tz-aware를 싫어하므로 tz 제거
    t = t.dt.tz_convert("UTC").dt.tz_localize(None)
    return t.dt.to_period(split).astype(str)


def quantile_threshold(s: pd.Series, q: float) -> float:
    s2 = pd.to_numeric(s, errors="coerce")
    return float(s2.quantile(q))


def safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series([np.nan] * len(df))


def short_hash(text: str, n: int = 10) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:n]


def build_tag(args) -> str:
    """
    파일명 길이/문자 문제를 피하기 위해:
    - 사람이 읽는 짧은 프리픽스 + 해시를 사용
    """
    core = {
        "h": args.h,
        "tp": args.tp,
        "sl": args.sl,
        "slice_q": args.slice_q,
        "dir_rule": args.dir_rule,
        "both_policy": args.both_policy,
        "require_eventret_match": int(args.require_eventret_match),
        "nohit_pnl": args.nohit_pnl,
        "singlepos_strict": int(args.singlepos_strict),
        "dedup_key": args.dedup_key or "",
        "dedup_keep": args.dedup_keep,
        "fee_bps": float(args.fee_bps),
        "slippage_bps": float(args.slippage_bps),
        "date_min": args.date_min or "",
        "date_max": args.date_max or "",
        # filters (only those explicitly set)
        "ret_over_ncvd_max": args.ret_over_ncvd_max,
        "event_vol_max": args.event_vol_max,
        "trade_count_max": args.trade_count_max,
        "event_ret_short_max": args.event_ret_short_max,
        "event_ret_long_min": args.event_ret_long_min,
        "ncvd_short_min": args.ncvd_short_min,
        "ncvd_long_max": args.ncvd_long_max,
    }
    blob = json.dumps(core, sort_keys=True, ensure_ascii=False)
    h = short_hash(blob, 12)
    # 사람이 읽을 수 있는 아주 짧은 prefix
    prefix = f"h{args.h}_tp{args.tp}_sl{args.sl}_q{args.slice_q}_{args.dir_rule}_{args.both_policy}_nohit{args.nohit_pnl}"
    return f"{prefix}__{h}"


def save_csv_safely(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


# -----------------------------
# Direction / Side
# -----------------------------
def add_side_from_rule(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    d = df.copy()
    if rule is None:
        raise ValueError("dir-rule is required (e.g., cvd_sign / ncvd_sign / event_ret_sign / hz_ret_sign / always_long / always_short)")

    rule = rule.lower().strip()
    if rule == "always_long":
        d["side"] = "long"
        return d
    if rule == "always_short":
        d["side"] = "short"
        return d

    col_map = {
        "cvd_sign": "cvd",
        "ncvd_sign": "ncvd",
        "event_ret_sign": "event_ret",
        "hz_ret_sign": "hz_ret",
    }
    if rule not in col_map:
        raise ValueError(f"Unknown dir-rule: {rule}. Use one of {list(col_map.keys())} or always_long/always_short.")

    v = pd.to_numeric(d[col_map[rule]], errors="coerce")
    side = np.empty(len(d), dtype=object)
    side[:] = None
    side[v > 0] = "long"
    side[v < 0] = "short"
    d["side"] = side
    d = d.dropna(subset=["side"]).reset_index(drop=True)
    return d


def apply_require_eventret_match(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    ev = pd.to_numeric(d["event_ret"], errors="coerce")
    side = d["side"].astype(str)
    ok = ((side == "long") & (ev > 0)) | ((side == "short") & (ev < 0))
    return d.loc[ok].reset_index(drop=True)


# -----------------------------
# TP/SL Hit Evaluation
# -----------------------------
def outcome_from_paths(
    side: str,
    tp: float,
    sl: float,
    high_path: List[float],
    low_path: List[float],
    both_policy: str,
) -> str:
    """
    high_path/low_path: entry 기준 누적수익률 path (list)
    tp/sl: +0.015, 0.012 (sl은 양수로 들어옴)
    """
    if high_path is None or low_path is None or len(high_path) == 0 or len(low_path) == 0:
        return "nohit"

    tp_level = float(tp)
    sl_level = float(sl)

    high = np.asarray(high_path, dtype=float)
    low = np.asarray(low_path, dtype=float)

    if side == "long":
        tp_hit = high >= (tp_level - 1e-15)
        sl_hit = low <= (-sl_level + 1e-15)
    else:
        tp_hit = low <= (-tp_level + 1e-15)
        sl_hit = high >= (sl_level - 1e-15)

    tp_i = np.where(tp_hit)[0]
    sl_i = np.where(sl_hit)[0]
    tp_first = int(tp_i[0]) if tp_i.size else None
    sl_first = int(sl_i[0]) if sl_i.size else None

    if tp_first is None and sl_first is None:
        return "nohit"
    if tp_first is not None and sl_first is None:
        return "tp"
    if tp_first is None and sl_first is not None:
        return "sl"

    # both hit
    if tp_first < sl_first:
        return "tp"
    if sl_first < tp_first:
        return "sl"

    # exact same bar
    if both_policy == "drop":
        return "both"
    if both_policy == "worst":
        return "sl"
    if both_policy == "best":
        return "tp"
    raise ValueError(f"Unknown both_policy={both_policy}")


def pnl_gross_from_outcome(outcome: str, side: str, tp: float, sl: float, nohit_pnl_mode: str, hz_close_path) -> float:
    """
    gross pnl (before fees/slippage)
    - tp: +tp
    - sl: -sl
    - nohit:
        - zero: 0
        - mtm_last: 마지막 close path를 MTM (short면 부호 반전)
    """
    if outcome == "tp":
        return float(tp)
    if outcome == "sl":
        return -float(sl)
    if outcome == "both":
        return np.nan

    # nohit
    if nohit_pnl_mode == "zero":
        return 0.0

    if nohit_pnl_mode == "mtm_last":
        last = None
        cp = parse_list_maybe(hz_close_path)
        if cp is not None and len(cp) > 0:
            last = float(cp[-1])
        # fallback: 없으면 0
        if last is None or np.isnan(last):
            last = 0.0
        return last if side == "long" else -last

    raise ValueError(f"Unknown nohit_pnl mode: {nohit_pnl_mode}")


# -----------------------------
# Dedup
# -----------------------------
def dedup_signals(df: pd.DataFrame, keys: List[str], keep: str) -> pd.DataFrame:
    d = df.copy()
    for k in keys:
        if k not in d.columns:
            raise ValueError(f"dedup-key column not found: {k}")

    if keep in ("first", "last"):
        d = d.sort_values(keys).drop_duplicates(keys, keep=keep).reset_index(drop=True)
        return d

    if keep in ("max_cv", "min_cv"):
        if "cv" not in d.columns:
            raise ValueError("Need column 'cv' for dedup-keep max_cv/min_cv")
        asc = (keep == "min_cv")
        d = d.sort_values(keys + ["cv"], ascending=[True] * len(keys) + [asc])
        d = d.drop_duplicates(keys, keep="last" if keep == "max_cv" else "first").reset_index(drop=True)
        return d

    raise ValueError(f"Unknown dedup-keep: {keep}")


# -----------------------------
# Filters
# -----------------------------
def apply_filters(df: pd.DataFrame, args) -> pd.DataFrame:
    d = df.copy()

    # event_ret thresholds
    if args.event_ret_short_max is not None:
        ev = safe_numeric(d, "event_ret")
        ok = (d["side"] != "short") | (ev <= float(args.event_ret_short_max))
        d = d.loc[ok]

    if args.event_ret_long_min is not None:
        ev = safe_numeric(d, "event_ret")
        ok = (d["side"] != "long") | (ev >= float(args.event_ret_long_min))
        d = d.loc[ok]

    # ncvd thresholds (사용자 의도: short는 더 음수(<=), long은 더 양수(>=))
    if args.ncvd_short_min is not None:
        v = safe_numeric(d, "ncvd")
        ok = (d["side"] != "short") | (v <= float(args.ncvd_short_min))
        d = d.loc[ok]

    if args.ncvd_long_max is not None:
        v = safe_numeric(d, "ncvd")
        ok = (d["side"] != "long") | (v >= float(args.ncvd_long_max))
        d = d.loc[ok]

    # global cuts
    if args.ret_over_ncvd_max is not None and "ret_over_ncvd" in d.columns:
        v = safe_numeric(d, "ret_over_ncvd")
        d = d.loc[v <= float(args.ret_over_ncvd_max)]

    if args.event_vol_max is not None and "event_volatility" in d.columns:
        v = safe_numeric(d, "event_volatility")
        d = d.loc[v <= float(args.event_vol_max)]

    if args.trade_count_max is not None and "trade_count" in d.columns:
        v = safe_numeric(d, "trade_count")
        d = d.loc[v <= float(args.trade_count_max)]

    return d.reset_index(drop=True)


# -----------------------------
# Simulation: single_position
# -----------------------------
def simulate_single_position(
    df: pd.DataFrame,
    fee_bps: float,
    slippage_bps: float,
    strict: bool,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    required = ["entry_ts", "side", "outcome", "pnl_gross"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    d = df.copy().sort_values("entry_ts").reset_index(drop=True)

    # active rows
    if strict:
        active = d["outcome"].isin(["tp", "sl", "nohit"])
    else:
        active = d["outcome"].isin(["tp", "sl"])

    dd = d.loc[active].copy().reset_index(drop=True)

    # costs (roundtrip)
    fee = float(fee_bps) / 10000.0
    slip = float(slippage_bps) / 10000.0
    roundtrip_cost = 2.0 * (fee + slip)

    dd["fee_bps"] = float(fee_bps)
    dd["slippage_bps"] = float(slippage_bps)
    dd["pnl_net"] = pd.to_numeric(dd["pnl_gross"], errors="coerce") - roundtrip_cost

    pnl = dd["pnl_net"].to_numpy(dtype=float)
    if pnl.size == 0:
        res = dict(trades=0, exp=0.0, sum=0.0, win=0.0, mdd=0.0)
        return res, dd

    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak
    mdd = float(drawdown.min()) if drawdown.size else 0.0

    res = dict(
        trades=int(pnl.size),
        exp=float(np.mean(pnl)),
        sum=float(np.sum(pnl)),
        win=float(np.mean(pnl > 0)),
        mdd=mdd,
    )
    return res, dd


# -----------------------------
# Outputs
# -----------------------------
def build_outputs(df: pd.DataFrame, args, out_dir: Path) -> None:
    ensure_dir(out_dir)

    tag = build_tag(args)

    # save run meta (repro)
    meta = vars(args).copy()
    meta_path = out_dir / f"run_meta__{tag}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # fee sensitivity
    fee_grid = [float(x.strip()) for x in args.fee_grid.split(",")] if args.fee_grid else [0, 1, 2, 5, 10]
    rows = []
    trades_preview = None

    for fb in fee_grid:
        res, trades = simulate_single_position(
            df,
            fee_bps=fb,
            slippage_bps=args.slippage_bps,
            strict=args.singlepos_strict,
        )
        rows.append(dict(fee_bps=fb, slippage_bps=float(args.slippage_bps), **res))
        if abs(fb - 0.0) < 1e-12:
            trades_preview = trades

    fee_df = pd.DataFrame(rows).sort_values("fee_bps").reset_index(drop=True)
    fee_path = out_dir / f"fee_sensitivity__singlepos__{tag}.csv"
    save_csv_safely(fee_df, fee_path)

    if trades_preview is not None:
        prev = trades_preview.copy()
        prev["period"] = to_period_str(prev["entry_ts"], args.time_split or "Q")
        prev_path = out_dir / f"trades_preview__singlepos__fee0bps__{tag}.csv"
        save_csv_safely(prev, prev_path)

    print("\n=== FEE SENSITIVITY (single_position) ===")
    print(fee_df.to_string(index=False))

    # time split (single fee)
    time_fee = float(args.fee_bps)
    resT, tradesT = simulate_single_position(
        df,
        fee_bps=time_fee,
        slippage_bps=args.slippage_bps,
        strict=args.singlepos_strict,
    )

    if len(tradesT) == 0:
        print("\n=== TIME SPLIT ===")
        print("(no trades)")
        return

    tradesT = tradesT.copy()
    tradesT["period"] = to_period_str(tradesT["entry_ts"], args.time_split or "Q")

    grp = tradesT.groupby("period", as_index=False).agg(
        fee_bps=("fee_bps", "first"),
        slippage_bps=("slippage_bps", "first"),
        trades=("pnl_net", "size"),
        exp=("pnl_net", "mean"),
        sum=("pnl_net", "sum"),
        win=("pnl_net", lambda x: float(np.mean(np.asarray(x) > 0))),
    )

    # mdd per period
    mdds = []
    for p, g in tradesT.groupby("period"):
        pnl = g["pnl_net"].to_numpy(dtype=float)
        if pnl.size == 0:
            mdds.append((p, 0.0))
            continue
        eq = np.cumsum(pnl)
        pk = np.maximum.accumulate(eq)
        dd = eq - pk
        mdds.append((p, float(dd.min())))
    mdd_df = pd.DataFrame(mdds, columns=["period", "mdd"])
    grp = grp.merge(mdd_df, on="period", how="left").sort_values("period").reset_index(drop=True)

    ts_path = out_dir / f"time_split__singlepos__fee{time_fee}bps__{tag}.csv"
    save_csv_safely(grp, ts_path)

    print(f"\n=== TIME SPLIT (fee={time_fee} bps, split={args.time_split or 'Q'}) ===")
    print(grp.to_string(index=False))

    # culprit analysis
    if args.culprit_period:
        c = tradesT.loc[tradesT["period"] == args.culprit_period].copy()
        if args.culprit_loss_only:
            c = c.loc[c["pnl_net"] < 0].copy()

        if len(c) == 0:
            print(f"\n[CULPRIT] period={args.culprit_period}: no rows (after loss_only={int(args.culprit_loss_only)})")
            return

        c["culprit"] = -c["pnl_net"]
        c = c.sort_values("culprit", ascending=False).head(int(args.culprit_topk)).reset_index(drop=True)

        cul_path = out_dir / f"culprit_top{int(args.culprit_topk)}__fee{time_fee}bps__{tag}__{args.culprit_period}.csv"
        save_csv_safely(c, cul_path)
        print(f"\nSaved -> {cul_path}")

        # feature summary
        feat_cols = [
            "cv", "cvd", "ncvd", "event_ret", "event_range", "event_volatility",
            "hz_ret", "hz_max_up", "hz_max_dn", "hz_vol",
        ]
        feat_cols = [x for x in feat_cols if x in tradesT.columns]
        feat = tradesT.loc[tradesT["period"] == args.culprit_period, feat_cols].apply(pd.to_numeric, errors="coerce")
        summary = feat.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).T.reset_index().rename(columns={"index": "feature"})
        sum_path = out_dir / f"feature_summary__fee{time_fee}bps__{tag}__{args.culprit_period}.csv"
        save_csv_safely(summary, sum_path)
        print(f"Saved -> {sum_path}")


# -----------------------------
# CLI
# -----------------------------
def build_argparser():
    ap = argparse.ArgumentParser(
        description="Event horizon both-hit diagnostics (scratch3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ap.add_argument("--in-parquet", required=True, type=str)
    ap.add_argument("--out-dir", required=True, type=str)

    ap.add_argument("--h", required=True, type=int)
    ap.add_argument("--tp", required=True, type=float)
    ap.add_argument("--sl", required=True, type=float)

    ap.add_argument("--slice-q", type=float, default=0.9)
    ap.add_argument("--dir-rule", type=str, default="cvd_sign")
    ap.add_argument("--both-policy", choices=["drop", "worst", "best"], default="drop")
    ap.add_argument("--require_eventret_match", action="store_true")

    ap.add_argument("--dedup-key", type=str, default=None)
    ap.add_argument("--dedup-keep", choices=["first", "last", "max_cv", "min_cv"], default="first")
    ap.add_argument("--singlepos-strict", action="store_true")

    ap.add_argument("--nohit-pnl", choices=["zero", "mtm_last"], default="zero")

    ap.add_argument("--fee-grid", type=str, default="0,1,2,5,10")
    ap.add_argument("--fee-bps", type=float, default=0.0)
    ap.add_argument("--slippage-bps", type=float, default=0.0)

    ap.add_argument("--time-split", choices=["Q", "M"], default="Q")
    ap.add_argument("--date-min", type=str, default=None)
    ap.add_argument("--date-max", type=str, default=None)

    # filters
    ap.add_argument("--ret-over-ncvd-max", type=float, default=None)
    ap.add_argument("--event-vol-max", type=float, default=None)
    ap.add_argument("--trade-count-max", type=float, default=None)

    ap.add_argument("--event-ret-short-max", type=float, default=None)
    ap.add_argument("--event-ret-long-min", type=float, default=None)
    ap.add_argument("--ncvd-short-min", type=float, default=None)
    ap.add_argument("--ncvd-long-max", type=float, default=None)

    # culprit
    ap.add_argument("--culprit-period", type=str, default=None)
    ap.add_argument("--culprit-topk", type=int, default=50)
    ap.add_argument("--culprit-loss-only", action="store_true")

    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()

    in_path = Path(args.in_parquet)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    df = pd.read_parquet(in_path)
    print(f"Loaded rows: {len(df)}")

    # h filter
    if "h" not in df.columns:
        raise ValueError("Input parquet missing required column: h")
    df = df.loc[df["h"] == args.h].copy().reset_index(drop=True)
    print(f"Filtered h={args.h}: rows={len(df)}")

    # entry_ts
    if "entry_ts" not in df.columns:
        raise ValueError("Input parquet missing required column: entry_ts")
    df["entry_ts"] = to_utc_datetime(df["entry_ts"])
    df = df.dropna(subset=["entry_ts"]).sort_values("entry_ts").reset_index(drop=True)

    # date window
    if args.date_min:
        tmin = pd.to_datetime(args.date_min, utc=True)
        before = len(df)
        df = df.loc[df["entry_ts"] >= tmin].copy().reset_index(drop=True)
        print(f"After date-min {args.date_min}: rows={len(df)} (dropped {before-len(df)})")

    if args.date_max:
        tmax = pd.to_datetime(args.date_max, utc=True)
        # inclusive end-day처럼 쓰고 싶으면 23:59:59를 넣어도 되지만,
        # 여기서는 사용자가 날짜를 정확히 넣는다고 가정하고 <= 그대로 적용
        before = len(df)
        df = df.loc[df["entry_ts"] <= tmax].copy().reset_index(drop=True)
        print(f"After date-max {args.date_max}: rows={len(df)} (dropped {before-len(df)})")

    # normalize paths
    for col in ["hz_high_path", "hz_low_path"]:
        if col not in df.columns:
            raise ValueError(f"Input parquet missing required column: {col}")
        df[col] = df[col].map(parse_list_maybe)

    if "hz_close_path" in df.columns:
        df["hz_close_path"] = df["hz_close_path"].map(parse_list_maybe)

    df = df.dropna(subset=["hz_high_path", "hz_low_path"]).reset_index(drop=True)
    print(f"After path normalization/filter: rows={len(df)}")

    # side
    df = add_side_from_rule(df, args.dir_rule)

    # top slice by cv
    if "cv" not in df.columns:
        raise ValueError("Input parquet missing required column: cv")
    cv_th = quantile_threshold(df["cv"], args.slice_q)
    top_df = df.loc[pd.to_numeric(df["cv"], errors="coerce") >= cv_th].copy().reset_index(drop=True)
    print(f"Top slice rows: {len(top_df)} (slice_q={args.slice_q}, cv_th={cv_th})")

    # require event_ret match
    if args.require_eventret_match:
        if "event_ret" not in top_df.columns:
            raise ValueError("require_eventret_match requires column: event_ret")
        before = len(top_df)
        top_df = apply_require_eventret_match(top_df)
        print(f"After require_eventret_match: rows={len(top_df)} (dropped {before-len(top_df)})")

    # outcome + pnl_gross
    outs = []
    pnls = []
    for r in top_df.itertuples(index=False):
        out = outcome_from_paths(
            side=str(r.side),
            tp=float(args.tp),
            sl=float(args.sl),
            high_path=r.hz_high_path,
            low_path=r.hz_low_path,
            both_policy=args.both_policy,
        )
        outs.append(out)
        pnl = pnl_gross_from_outcome(
            outcome=out,
            side=str(r.side),
            tp=float(args.tp),
            sl=float(args.sl),
            nohit_pnl_mode=args.nohit_pnl,
            hz_close_path=getattr(r, "hz_close_path", None),
        )
        pnls.append(pnl)

    top_df["outcome"] = outs
    top_df["pnl_gross"] = pnls

    # drop both if policy=drop
    if args.both_policy == "drop":
        before = len(top_df)
        top_df = top_df.loc[top_df["outcome"] != "both"].copy().reset_index(drop=True)
        top_df = top_df.dropna(subset=["pnl_gross"]).reset_index(drop=True)
        # print(f"After both_policy=drop: rows={len(top_df)} (dropped {before-len(top_df)})")

    # dedup
    if args.dedup_key:
        keys = [k.strip() for k in args.dedup_key.split(",") if k.strip()]
        before = len(top_df)
        top_df = dedup_signals(top_df, keys=keys, keep=args.dedup_keep)
        print(f"After dedup_signals({','.join(keys)}, keep={args.dedup_keep}): rows={len(top_df)} (dropped {before-len(top_df)})")

    # filters
    before = len(top_df)
    top_df = apply_filters(top_df, args)
    print(f"After filters: rows={len(top_df)} (dropped {before-len(top_df)})")

    # outputs
    build_outputs(top_df, args, out_dir)

if __name__ == "__main__":
    main()
