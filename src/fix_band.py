from pathlib import Path
import re

import numpy as np
import pandas as pd

grid_re = re.compile(
    r"tf=(?P<tf>[^_]+)__w=(?P<w>\d+)__q=(?P<q>\d+\.\d+)__m=(?P<m>\d+)__g=(?P<g>\d+)__h=(?P<h>\d+)"

)

def read_report_csv(path: Path) -> pd.DataFrame | None:
    # 1) read csv files
    try:
        # 1-1) check header
        with path.open("r", errors="ignore") as f:
            header = f.readline()
            if not header or not header.strip():
                print(f"[SKIP] empty header line: {path}")
                return None
        # 1-2) get csv
        report_df = pd.read_csv(path)

        # 1-3) check column
        if report_df.empty:
            print(f"[SKIP] empty report_df: {path}")
            return None

        return report_df

    except pd.errors.EmptyDataError:
        print(f"[SKIP] EmptyDataError: {path}")
        return None
    except Exception as e:
        print(f"[SKIP] read error: {path}")
        return None



# ===== main =====
# run_dir = Path(r"D:\data1\profile-regime\experiments\binance\futures_um\symbol=BTCUSDT\run=20260105_163204_256")
run_dir = Path(r"D:\data\profile-regime\experiments\binance\futures_um\symbol=BTCUSDT\run=20260106_023135_536")
path_list = list(run_dir.rglob("atz_eval_report_all.csv"))
if not path_list:
    raise Exception("No atz_eval_report_all.csv found")

report_df_list: list[pd.DataFrame] = []
skipped = 0

for path in path_list:
    # read csv as df
    report_df = read_report_csv(path)
    if report_df is None:
        skipped += 1
        continue

    # parse grid
    grid_dict = grid_re.search(str(path)).groupdict()
    if not grid_dict:
        skipped += 1
        continue

    # add column
    for k,v in grid_dict.items():
        report_df[k] = v

    # append to df list
    report_df_list.append(report_df)

if not report_df_list:
    raise SystemExit("All report files were skipped (no usable data).")

all_report_df = pd.concat(report_df_list, ignore_index=True)
print(f"Loaded files: {len(report_df_list)}    skipped: {skipped}")
print(f"Loaded rows: {len(all_report_df):,}")

# change to numeric
num_cols = ['w', 'q', 'm', 'g', 'h']
all_report_df[num_cols] = all_report_df[num_cols].apply(pd.to_numeric)

target_cols = ['ks_d', 'cliffs_delta', 'diff_p90', 'diff_p95']

metrics = all_report_df['metric'].dropna().unique()
report_per_metric_list = []
for metric in metrics:
    report_per_metric = all_report_df[all_report_df['metric'] == metric].copy()

    report_per_metric["abs_cliffs"] = report_per_metric["cliffs_delta"].abs()

    # get rank
    A = report_per_metric["diff_p95"].rank(method='average', pct=True, ascending=False)
    B = report_per_metric["ks_d"].rank(method='average', pct=True, ascending=False)
    C = report_per_metric["abs_cliffs"].rank(method='average', pct=True, ascending=False)

    report_per_metric["score_rank"] = 0.65*A + 0.20*B + 0.15*C
    report_per_metric_list.append(report_per_metric)

all_report_df = pd.concat(report_per_metric_list, ignore_index=True)

# minimum required filter
all_report_df = all_report_df[all_report_df["n_atz"] >= 150]
all_report_df = all_report_df[all_report_df["diff_p95"] > 0.003]
all_report_df = all_report_df[all_report_df["ks_d"] >= 0.2]
all_report_df = all_report_df[all_report_df["abs_cliffs"] >= 0.2]

# quick check
print("\nScore sample (top 20 by score_rank for metric=range):")
range_df = all_report_df[all_report_df["metric"] == "range"].copy()
if not range_df.empty:
    show_cols = [
        "tf", "w", "q", "m", "g", "h",
        "n_atz", "diff_p95", "ks_d", "cliffs_delta", "score_rank"
    ]
    print(range_df.sort_values("score_rank", ascending=False)[show_cols].head(20).to_string(index=False))
else:
    print("No rows with metric=range")