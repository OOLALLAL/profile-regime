from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

# -----------------------------
# config
# -----------------------------
grid_re = re.compile(
    r"tf=(?P<tf>[^_]+)__w=(?P<w>\d+)__q=(?P<q>\d+\.\d+)__m=(?P<m>\d+)__g=(?P<g>\d+)__h=(?P<h>\d+)"
)

# 필수 컬럼 (최소)
REQUIRED_COLS = {"metric", "n_atz", "diff_p95", "ks_d", "cliffs_delta"}

# -----------------------------
# helpers
# -----------------------------
def safe_read_report_csv(path: Path) -> pd.DataFrame | None:
    """깨진/빈 CSV를 100% 안전하게 스킵. 성공하면 DF 반환, 실패하면 None."""
    try:
        # 1) 텍스트로 첫 줄 확인 (헤더 유무)
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            first = f.readline()
        if not first or not first.strip():
            print(f"[SKIP] empty header line: {path}")
            return None

        # 2) read_csv 자체가 EmptyDataError 등 낼 수 있으니 try
        df = pd.read_csv(path)

        # 3) 컬럼 검증
        if df.empty:
            # empty DF는 스킵(혹은 유지해도 되지만, 지금 목적상 스킵이 안전)
            print(f"[SKIP] df empty: {path}")
            return None

        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            print(f"[SKIP] missing cols {sorted(missing)}: {path}")
            return None

        return df

    except pd.errors.EmptyDataError:
        print(f"[SKIP] EmptyDataError: {path}")
        return None
    except Exception as e:
        print(f"[SKIP] read error: {path} -> {type(e).__name__}: {e}")
        return None


def parse_grid_from_path(path: Path) -> dict | None:
    m = grid_re.search(str(path))
    if not m:
        # grid_id가 경로에 없으면 나중에 합칠 때 골치 아파서 스킵하는 게 안전
        print(f"[SKIP] grid pattern not found in path: {path}")
        return None
    return m.groupdict()


def rank01(s: pd.Series, ascending: bool = True) -> pd.Series:
    """0~1 랭크 스케일. (큰 값이 좋으면 ascending=True 유지)"""
    # rank(pct=True)는 동일값 처리 등에서 안정적임
    r = s.rank(method="average", pct=True, ascending=ascending)
    # NaN은 그대로 두기
    return r


# -----------------------------
# main
# -----------------------------
run_dir = Path(r"D:\data\profile-regime\experiments\binance\futures_um\symbol=BTCUSDT\run=20260106_023135_536")
path_list = list(run_dir.rglob("atz_eval_report_all.csv"))
if not path_list:
    raise SystemExit("No atz_eval_report_all.csv found")

dfs: list[pd.DataFrame] = []
skipped = 0

for p in path_list:
    df = safe_read_report_csv(p)
    if df is None:
        skipped += 1
        continue

    grid = parse_grid_from_path(p)
    if grid is None:
        skipped += 1
        continue

    # grid 파라미터 컬럼 추가
    for k, v in grid.items():
        df[k] = v

    # numeric 변환 (안되면 NaN)
    for col in ["w", "q", "m", "g", "h"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # n_atz numeric
    df["n_atz"] = pd.to_numeric(df["n_atz"], errors="coerce")

    # 필요한 수치 컬럼들도 numeric
    for col in ["diff_p95", "ks_d", "cliffs_delta"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    dfs.append(df)

if not dfs:
    raise SystemExit("All report files were skipped (no usable data).")

all_report_df = pd.concat(dfs, ignore_index=True)
print(f"Loaded files: {len(dfs)}  skipped: {skipped}")
print(f"Loaded rows: {len(all_report_df):,}")

# -----------------------------
# rank score (metric별로)
# -----------------------------
# score는 'range'로만 보고 싶다면 여기서 필터해도 됨.
# (요청은 'rank score'까지만이라 전체 metric에 대해 생성)
def add_rank_score_per_metric(df: pd.DataFrame) -> pd.DataFrame:
    out_parts = []
    for metric, gdf in df.groupby("metric", dropna=True):
        gdf = gdf.copy()

        # 핵심: cliffs는 abs로
        gdf["abs_cliffs"] = gdf["cliffs_delta"].abs()

        # 랭크 (큰 값이 좋다)
        A = rank01(gdf["diff_p95"], ascending=True)
        B = rank01(gdf["ks_d"], ascending=True)
        C = rank01(gdf["abs_cliffs"], ascending=True)

        # 가중합 (추천 비율)
        gdf["score_rank"] = 0.60 * A + 0.25 * B + 0.15 * C

        out_parts.append(gdf)

    return pd.concat(out_parts, ignore_index=True) if out_parts else df.assign(score_rank=np.nan)


all_report_df = add_rank_score_per_metric(all_report_df)

# 빠르게 확인
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
