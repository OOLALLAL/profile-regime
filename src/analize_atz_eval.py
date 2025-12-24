import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

def _summ(a: np.ndarray) -> dict:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return dict(n=0, mean=np.nan, median=np.nan, p90=np.nan, p95=np.nan, std=np.nan)
    return dict(
        n=int(a.size),
        mean=float(a.mean()),
        median=float(np.quantile(a, 0.50)),
        p90=float(np.quantile(a, 0.90)),
        p95=float(np.quantile(a, 0.95)),
        std=float(a.std(ddof=1)) if a.size > 1 else 0.0,
    )
def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return np.nan
    y_sorted = np.sort(y)
    lt = np.searchsorted(y_sorted, x, side="left") # less than
    le = np.searchsorted(y_sorted, x, side="right") # less or equal
    gt = y.size - le # greater than
    return float((lt.sum() - gt.sum()) / (x.size * y.size))

def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return np.nan
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    pooled = np.sqrt(((x.size - 1) * vx + (y.size - 1) * vy) / (x.size + y.size - 2))
    return float((x.mean() - y.mean()) / pooled) if pooled > 0 else 0.0

def one_report(df: pd.DataFrame, metrics: list[str], atz_label: str, base_label:str, scope: dict) -> pd.DataFrame:
    g = df["group"].astype(str)
    out = []

    for m in metrics:
        if m not in df.columns:
            continue

        x = df.loc[g == atz_label ,m].to_numpy()
        y = df.loc[g == base_label, m].to_numpy()

        sx , sy = _summ(x), _summ(y)
        if sx["n"] == 0 or sy["n"] == 0:
            continue

        ks = stats.ks_2samp(x, y, alternative="two-sided", method="auto")
        mw = stats.mannwhitneyu(x, y, alternative="two-sided", method="auto")

        row = {
            **scope,
            "metric": m,
            "n_atz": sx["n"],
            "n_base": sy["n"],

            "base_mean": sy["mean"],
            "base_median": sy["median"],
            "base_p90": sy["p90"],
            "base_p95": sy["p95"],

            "atz_mean": sx["mean"],
            "atz_median": sx["median"],
            "atz_p90": sx["p90"],
            "atz_p95": sx["p95"],

            "diff_mean": sx["mean"] - sy["mean"],
            "diff_median": sx["median"] - sy["median"],
            "diff_p90": sx["p90"] - sy["p90"],
            "diff_p95": sx["p95"] - sy["p95"],

            "ks_d": float(ks.statistic),
            "ks_p": float(ks.pvalue),

            "mw_u": float(mw.statistic),
            "mw_p": float(mw.pvalue),

            "cliffs_delta": cliffs_delta(x, y),
            "cohens_d": cohens_d(x, y),
        }
        out.append(row)

    return pd.DataFrame(out)

@dataclass
class Args:
    in_events: Path
    out_dir: Path
    metrics: str
    atz_label: str
    base_label: str
    by_symbol: bool
    by_date: bool
    log_level: str

def parse_args() -> Args:
    p = argparse.ArgumentParser("Analyze ATZ vs baseline")
    p.add_argument("--in-events", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--metrics", type=str, default="mfe,mae,range")
    p.add_argument("--atz-label", type=str, default="atz")
    p.add_argument("--base-label", type=str, default="baseline")
    p.add_argument("--by-date", action="store_true")
    p.add_argument("--log-level", default="INFO")
    ns = p.parse_args()
    return Args(**vars(ns))

def run(args: Args) -> None:
    setup_logging(args.log_level)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.in_events)
    if df.empty:
        raise SystemExit("Empty input")

    if "group" not in df.columns:
        raise SystemExit("Missing 'group' column in input")

    metrics = [s.strip() for s in args.metrics.split(",") if s.strip()]

    rep_all = one_report(df, metrics, args.atz_label, args.base_label, scope={"scope": "all"})
    rep_all.to_parquet(args.out_dir / "atz_eval_report_all.parquet", index=False)
    rep_all.to_csv(args.out_dir / "atz_eval_report_all.csv", index=False)
    logger.info("OK report_all rows=%d", len(rep_all))

    if args.by_date and "date" in df.columns:
        parts = []
        for dt, gdf in df.groupby("date"):
            parts.append(one_report(gdf, metrics, args.atz_label, args.base_label, scope={"scope": "date", "date": dt}))
        rep_dt = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        rep_dt.to_parquet(args.out_dir / "atz_eval_report_by_date.parquet", index=False)
        rep_dt.to_csv(args.out_dir / "atz_eval_report_by_date.csv", index=False)
        logger.info("OK report_by_date rows=%d", len(rep_dt))

    if not rep_all.empty:
        top = rep_all.sort_values("ks_d", ascending=False).head(10)
        logger.info("Top by KS D:\n%s", top[["metric","ks_d","ks_p","cliffs_delta","diff_p95","diff_p90","diff_median","diff_mean"]].to_string(index=False))

if __name__ == "__main__":
    run(parse_args())