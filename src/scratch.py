import argparse
import datetime as dt
import logging
import subprocess
import sys
from pathlib import Path

# -------------------------
# logging
# -------------------------
def setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

logger = logging.getLogger("run")


def run_cmd(cmd: list[str]):
    logger.info("CMD  %s", " ".join(cmd))
    subprocess.check_call(cmd)


# -------------------------
# args
# -------------------------
def parse_args():
    p = argparse.ArgumentParser("Full ATZ pipeline runner")

    # ===== download =====
    p.add_argument("--symbol", required=True, help="e.g. BTCUSDT")
    p.add_argument("--start", required=True, help="YYYYMMDD")
    p.add_argument("--end", required=True, help="YYYYMMDD")

    # ===== common dirs =====
    p.add_argument("--data-root", type=Path, required=True,
                   help="BASE DATA DIR (e.g. D:/data/profile-regime)")

    # ===== tf / atz =====
    p.add_argument("--tf", default="15min")
    p.add_argument("--atz-window", type=int, default=8)
    p.add_argument("--atz-q", type=float, default=0.90)
    p.add_argument("--atz-min-bars", type=int, default=2)
    p.add_argument("--atz-merge-gap", type=int, default=1)

    # ===== eval =====
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--baseline-ratio", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=7)

    # ===== analyze =====
    p.add_argument("--metrics", default="mfe,mae,range")
    p.add_argument("--by-date", action="store_true")

    # ===== behavior =====
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log-level", default="INFO")

    return p.parse_args()


# -------------------------
# main
# -------------------------
def main():
    args = parse_args()
    setup_logging(args.log_level)

    py = sys.executable

    # -------------------------
    # directory convention (PLACEHOLDER)
    # -------------------------
    base = args.data_root

    zips_dir = base / "zips"                    # ← download 결과
    trades_dir = base / "aggTrades_parquet"     # ← build_aggtrades_parquet
    bars_1m_dir = base / "bars_1m"              # ← build_1m_bars
    features_dir = base / "features_tf"         # ← build_tf_features
    atz_dir = base / "events" / "atz"            # ← build_atz_events
    eval_dir = base / "eval" / f"symbol={args.symbol}"
    report_dir = base / "report" / f"symbol={args.symbol}"

    eval_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # RUN ID (for logs only)
    # -------------------------
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("RUN %s | %s %s~%s", run_id, args.symbol, args.start, args.end)

    # ============================================================
    # 1) download_um_aggtrades_daily.py
    # ============================================================
    run_cmd([
        py, "src/download_um_aggtrades_daily.py",
        "--symbol", args.symbol,
        "--start", args.start,
        "--end", args.end,
        "--out-dir", str(base),
    ])

    # ============================================================
    # 2) build_aggtrades_parquet.py
    # ============================================================
    cmd2 = [
        py, "src/build_aggtrades_parquet.py",
        "--zips-dir", str(zips_dir),
        "--out-dir", str(trades_dir),
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
        "--out-dir", str(bars_1m_dir),
    ]
    if args.overwrite:
        cmd3.append("--overwrite")
    run_cmd(cmd3)

    # ============================================================
    # 4) build_tf_features.py
    # ============================================================
    cmd4 = [
        py, "src/build_tf_features.py",
        "--bars-1m-dir", str(bars_1m_dir),
        "--out-dir", str(features_dir),
        "--tf", args.tf,
        "--r-norm", "range",
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
        "--tf", args.tf,
        "--window", str(args.atz_window),
        "--q", str(args.atz_q),
        "--min-bars", str(args.atz_min_bars),
        "--merge-gap", str(args.atz_merge_gap),
    ]
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
    ]
    run_cmd(cmd6)

    # ============================================================
    # 7) analize_atz_eval.py
    # ============================================================
    cmd7 = [
        py, "src/analize_atz_eval.py",
        "--in-events", str(eval_dir / "atz_eval_events.parquet"),
        "--out-dir", str(report_dir),
        "--metrics", args.metrics,
    ]
    if args.by_date:
        cmd7.append("--by-date")
    run_cmd(cmd7)

    logger.info("PIPELINE DONE")
    logger.info("eval   -> %s", eval_dir)
    logger.info("report -> %s", report_dir)


if __name__ == "__main__":
    main()
