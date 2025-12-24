# NOTE: run this from project root: python src/run.py ...
import argparse
import logging
import sys
import subprocess
from typing import List

# -------------------------
# logging
# -------------------------
def setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

logger = logging.getLogger(__name__)

def run_cmd(cmd: List[str]):
    logger.info("CMD %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

# -------------------------
# args
# -------------------------
def parse_args():
    p = argparse.ArgumentParser("Full pipeline runner")

    # ===== download =====
    p.add_argument("--symbol")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--sleep-sec")

    # ===== dir =====
    p.add_argument("--data-root")

    # ===== tf / atz =====
    p.add_argument("--tf", type=str, required=True)
    p.add_argument("--r-norm", type=str, default="range", choices=["range", "vol_tanh"])
    p.add_argument("--vol-window", type=int, default=96)
    p.add_argument("--eps", type=float, default=1e-12)
    p.add_argument("--window", type=int, default=96)
    p.add_argument("--q", type=float, default=0.90)
    p.add_argument("--min-bars", type=int, default=2)
    p.add_argument("--merge-gap", type=int, default=1)
    p.add_argument("--no-cv", action="store_true")
    p.add_argument("--no-trade-count", action="store_true")

    # ===== eval =====
    p.add_argument("--horizon", type=int, default=8, help="future bars after ATZ end")
    p.add_argument("--baseline-ratio", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=7)

    # ===== analyze =====
    p.add_argument("--metrics", type=str, default="mfe,mae,range")
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

    # ===== directory convention =====
    root_dir = args.data_root

    zips_dir = root_dir
    trades_dir = root_dir
    bars_dir = root_dir
    features_dir = root_dir
    atz_dir = root_dir
    eval_dir = root_dir
    report_dir = root_dir


    # ===== RUN ID (only for logs) =====

    # ===== 1) download_um_aggtrades_daily.py =====
    # ===== 2) build_aggtrades_parquet.py =====
    # ===== 3) build_1m_bars.py =====
    # ===== 4) build_tf_features.py =====
    # ===== 5) build_atz_events.py =====
    # ===== 6) evaluate_atz.py =====
    # ===== 7) analize_atz_eval.py =====

    logger.info("PIPELINE DONE")
    logger.info("eval   -> %s", )
    logger.info("report -> %s", )

if __name__ == "__main__":
    main()