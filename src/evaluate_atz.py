import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

FEATURE_RE = re.compile(
    r"^(?P<symbol>[A-Z0-9]+)-features-(?P<tf>[^-]+)-(?P<date>\d{4}-\d{2}-\d{2})\.parquet$"
)
ATZ_RE = re.compile(
    r"^(?P<symbol>[A-Z0-9]+)-atz-(?P<tf>[^-]+)-(?P<date>\d{4}-\d{2}-\d{2})\.parquet$"
)

def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

def write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(out_path)

def future_metrics(df: pd.DataFrame, idx: int, horizon: int) -> dict:
    entry = df.loc[idx, "close"]
    end = idx + horizon
    hi = df.loc[idx+1:end, "high"].max()
    lo = df.loc[idx+1:end, "low"].min()

    return {
        "mfe": (hi - entry) / entry,
        "mae": (entry - lo) / entry,
        "range": (hi - lo) / entry,
    }
@dataclass
class Args:
    features_dir: Path
    atz_dir: Path
    out_dir: Path

    horizon: int
    baseline_ratio: float
    seed: int

    overwrite: bool
    log_level: str

def parse_args():
    p = argparse.ArgumentParser("Evaluate ATZ vs baseline")
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--atz-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)

    p.add_argument("--horizon", type=int, default=8, help="future bars after ATZ end")
    p.add_argument("--baseline-ratio", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=7)

    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log-level", default="INFO")
    ns = p.parse_args()
    return Args(**vars(ns))

def run(args: Args):
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    feat_files = sorted(args.features_dir.glob("*.parquet"))
    atz_files = sorted(args.atz_dir.glob("*.parquet"))

    fmap = {}
    for fp in feat_files:
        match = FEATURE_RE.match(fp.name)
        if match:
            meta = match.groupdict()
            fmap[(meta["symbol"], meta["tf"], meta["date"])] = fp

    rows = []
    for ap in atz_files:
        match = ATZ_RE.match(ap.name)
        if match:
            meta = match.groupdict()
            key = (meta["symbol"], meta["tf"], meta["date"])

            if key not in fmap:
                continue

            df = pd.read_parquet(fmap[key])
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
            df = df.sort_values("ts").reset_index(drop=True)

            ev = pd.read_parquet(ap)
            # ATZ
            for r in ev.itertuples(index=False):
                idx = int(r.end_idx)
                if idx + args.horizon >= len(df):
                    continue
                m = future_metrics(df, idx, args.horizon)
                rows.append(
                    {
                        "group": "atz",
                        "symbol": meta["symbol"],
                        "tf": meta["tf"],
                        "date": meta["date"],
                        **m,
                    }
                )
            # BASELINE
            blocked = np.zeros(len(df), dtype=bool)
            for r in ev.itertuples(index=False):
                blocked[int(r.start_idx):int(r.end_idx)+1] = True

            valid = np.where(~blocked)[0]
            valid = valid[valid + args.horizon < len(df)]

            n_base = min(int(len(ev) * args.baseline_ratio), len(valid))

            base_idx = rng.choice(valid, size=n_base, replace=False)

            for idx in base_idx:
                m = future_metrics(df, idx, args.horizon)
                rows.append(
                    {
                        "group": "baseline",
                        "symbol": meta["symbol"],
                        "tf": meta["tf"],
                        "date": meta["date"],
                        **m
                    }
                )
    df_all = pd.DataFrame(rows)

    summary = df_all.groupby("group")[["mfe","mae","range"]].describe(percentiles=[0.5,0.9,0.95])

    write_parquet(df_all, args.out_dir / "atz_eval_events.parquet")
    write_parquet(summary.reset_index(), args.out_dir / "atz_eval_summary.parquet")

    logger.info("OK events=%d", len(df_all))
    logger.info("OK summary saved")

if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.log_level)

    run(args)
