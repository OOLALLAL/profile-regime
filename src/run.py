# NOTE: run this from project root: python src/run.py ...
import argparse
import logging
import sys
import subprocess
from pathlib import Path

# -------------------------
# logging
# -------------------------
def setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

logger = logging.getLogger(__name__)

def run_cmd(cmd: list[str]):
    logger.info("CMD %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

# -------------------------
# args
# -------------------------
def parse_args():
    p = argparse.ArgumentParser("Full pipeline runner")

    # ===== download =====
    p.add_argument("--symbol", required=True, help="e.g. BTCUSDT")
    p.add_argument("--start", required=True, help="YYYYMMDD")
    p.add_argument("--end", required=True, help="YYYYMMDD")
    p.add_argument("--sleep-sec", type=float, default=0.2, help="Sleep between requests to be polite")

    # ===== dir =====
    p.add_argument("--data-root", type=Path, required=True, help="D:\data\profile-regime")

    # ===== tf / atz =====
    p.add_argument("--tf", type=str, required=True)
    p.add_argument("--r-norm", type=str, default="range", choices=["range", "vol_tanh"])
    p.add_argument("--vol-window", type=int, default=96, help="rolling window for vol_tanh (in TF bars)")
    p.add_argument("--eps", type=float, default=1e-12)

    p.add_argument("--window", type=int, default=96, help="rolling window in TF bars (96=1 day for 15m)")
    p.add_argument("--q", type=float, default=0.90, help="quantile for activity threshold")
    p.add_argument("--min-bars", type=int, default=2, help="min consecutive bars for an ATZ event")
    p.add_argument("--merge-gap", type=int, default=1, help="merge events separated by <= gap bars")
    p.add_argument("--no-cv", action="store_true", help="disable cv condition")
    p.add_argument("--no-trade-count", action="store_true", help="disable trade_count condition")

    # ===== eval =====
    p.add_argument("--horizon", type=int, required=True, help="future bars after ATZ end")
    p.add_argument("--baseline-ratio", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=7)

    # ===== analyze =====
    p.add_argument("--metrics", type=str, default="mfe,mae,range")
    p.add_argument("--atz-label", type=str, default="atz")
    p.add_argument("--base-label", type=str, default="baseline")
    p.add_argument("--by-date", action="store_true")

    # ===== behavior =====
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log-level", default="INFO")

    # ===== id =====
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument("--grid-id", type=str, required=True)

    return p.parse_args()

# -------------------------
# main
# -------------------------
def main():
    args = parse_args()
    setup_logging(args.log_level)

    py = sys.executable

    # ===== directory convention =====
    root_dir = args.data_root

    zips_dir = root_dir/"raw"/"binance"/"futures_um"/"aggTrades"/f"symbol={args.symbol}"

    cache_dir = root_dir/"cache"/"binance"/"futures_um"/f"symbol={args.symbol}"
    trades_dir = cache_dir/"aggTrades"
    bars_dir = cache_dir/"bars_1m"
    features_dir = cache_dir/"features_tf"/f"tf={args.tf}"

    exp_dir = root_dir/"experiments"/"binance"/"futures_um"/f"symbol={args.symbol}"/f"run={args.run_id}"/f"grid={args.grid_id}"
    atz_dir = exp_dir / "events" / "atz"
    eval_dir = exp_dir / "eval" / "atz_events"
    report_dir = exp_dir / "report" / "atz_eval"

    # ============================================================
    # 1) download_um_aggtrades_daily.py
    # ============================================================
    cmd1 = [
        py, "src/download_um_aggtrades_daily.py",
        "--symbol", args.symbol,
        "--start", args.start,
        "--end", args.end,
        "--out-dir", str(zips_dir),
        "--log-level", args.log_level,
    ]
    run_cmd(cmd1)

    # ============================================================
    # 2) build_aggtrades_parquet.py
    # ============================================================
    cmd2 = [
        py, "src/build_aggtrades_parquet.py",
        "--zips-dir", str(zips_dir),
        "--out-dir", str(trades_dir),
        "--log-level", args.log_level,
    ]
    if args.overwrite:
        cmd2.append("--overwrite")
    run_cmd(cmd2)

    # ============================================================
    # 3) build_1m_bars.py
    # ============================================================
    cmd3 = [
        py, "src/build_1m_bars.py",
        "--trades-dir", str(trades_dir),
        "--out-dir", str(bars_dir),
        "--log-level", args.log_level,
    ]
    if args.overwrite:
        cmd3.append("--overwrite")
    run_cmd(cmd3)

    # ============================================================
    # 4) build_tf_features.py
    # ============================================================
    cmd4 = [
        py, "src/build_tf_features.py",
        "--bars-1m-dir", str(bars_dir),
        "--out-dir", str(features_dir),
        "--tf", args.tf,
        "--r-norm", args.r_norm,
        "--vol-window", str(args.vol_window),
        "--eps", str(args.eps),
        "--log-level", args.log_level,
    ]
    run_cmd(cmd4)

    # ============================================================
    # 5) build_atz_events.py
    # ============================================================
    cmd5 = [
        py, "src/build_atz_events.py",
        "--features-dir", str(features_dir),
        "--out-dir", str(atz_dir),
        "--symbol", args.symbol,
        "--tf", str(args.tf),
        "--window", str(args.window),
        "--q", str(args.q),
        "--min-bars", str(args.min_bars),
        "--merge-gap", str(args.merge_gap),
        "--log-level", args.log_level,
    ]
    if args.overwrite:
        cmd5.append("--overwrite")
    if args.no_cv:
        cmd5.append("--no-cv")
    if args.no_trade_count:
        cmd5.append("--no-trade-count")
    run_cmd(cmd5)

    # ============================================================
    # 6) evaluate_atz.py
    # ============================================================
    cmd6 = [
        py, "src/evaluate_atz.py",
        "--features-dir", str(features_dir),
        "--atz-dir", str(atz_dir),
        "--out-dir", str(eval_dir),
        "--horizon", str(args.horizon),
        "--baseline-ratio", str(args.baseline_ratio),
        "--seed", str(args.seed),
        "--log-level", args.log_level,
    ]
    if args.overwrite:
        cmd6.append("--overwrite")
    run_cmd(cmd6)

    # ============================================================
    # 7) analize_atz_eval.py
    # ============================================================
    cmd7 = [
        py, "src/analize_atz_eval.py",
        "--in-events", str(eval_dir),
        "--out-dir", str(report_dir),
        "--metrics", str(args.metrics),
        "--atz-label", str(args.atz_label),
        "--base-label", str(args.base_label),
        "--log-level", args.log_level,
    ]
    if args.by_date:
        cmd7.append("--by-date")
    run_cmd(cmd7)

    logger.info("PIPELINE DONE")
    logger.info("eval   -> %s", eval_dir)
    logger.info("report -> %s", report_dir)

if __name__ == "__main__":
    main()