import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# config
# =========================
DATA_PATH = Path(r"D:\data\profile-regime\experiments\binance\futures_um\symbol=BTCUSDT\event_features\event_horizon_features.parquet")

# band filter (원하면 켜고/끄기)
USE_BAND_FILTER = True
BAND = {
    "tf": ["15min"],
    "w": [96],
    "m": [2, 4],
    "g": [0, 1],
}

# heatmap bins
NX = 10  # ncvd bins
NY = 10  # ret_over_ncvd bins

# minimum samples per cell to trust (표본 적은 칸 무시용)
MIN_COUNT_PER_CELL = 30

# =========================
# load
# =========================
df = pd.read_parquet(DATA_PATH)

# 필수 컬럼 체크
need = ["ncvd", "ret_over_ncvd", "hz_ret"]
missing = [c for c in need if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# =========================
# optional band filter
# =========================
if USE_BAND_FILTER:
    mask = (
        df["tf"].isin(BAND["tf"])
        & df["w"].isin(BAND["w"])
        & df["m"].isin(BAND["m"])
        & df["g"].isin(BAND["g"])
    )
    df = df.loc[mask].copy()

# =========================
# clean + target
# =========================
# 무한/결측 제거
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["ncvd", "ret_over_ncvd", "hz_ret"]).copy()

# 승률 타겟: horizon 수익이 양수인가?
df["win"] = (df["hz_ret"] > 0).astype(int)

print("Rows after filter:", len(df))

# =========================
# quantile binning
# =========================
# qcut은 동일값이 많으면 bin이 깨질 수 있어서 duplicates='drop' 사용
df["bin_x"] = pd.qcut(df["ncvd"], q=NX, duplicates="drop")
df["bin_y"] = pd.qcut(df["ret_over_ncvd"], q=NY, duplicates="drop")

# 실제 bin 개수(중복값 때문에 줄어들 수 있음)
x_bins = df["bin_x"].cat.categories
y_bins = df["bin_y"].cat.categories
print("Actual bins:", len(x_bins), "x", len(y_bins))

# =========================
# conditional win-rate table
# =========================
# 승률(평균) + 샘플 수
winrate = df.pivot_table(index="bin_y", columns="bin_x", values="win", aggfunc="mean")
counts  = df.pivot_table(index="bin_y", columns="bin_x", values="win", aggfunc="size")

# 표본 작은 칸은 NaN 처리 (시각화에서 비워짐)
winrate_masked = winrate.where(counts >= MIN_COUNT_PER_CELL)

# =========================
# plot heatmap (matplotlib)
# =========================
fig, ax = plt.subplots(figsize=(12, 9))

# imshow용 배열 (y가 위->아래로 커지게 하려면 origin="lower")
im = ax.imshow(winrate_masked.to_numpy(), origin="lower", aspect="auto")

# colorbar
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("P(hz_ret > 0)")

# tick labels: bin 구간을 보기 좋게 짧게 표시
def fmt_interval(iv):
    # iv: pandas Interval
    return f"{iv.left:.2f}~{iv.right:.2f}"

ax.set_xticks(np.arange(winrate_masked.shape[1]))
ax.set_yticks(np.arange(winrate_masked.shape[0]))
ax.set_xticklabels([fmt_interval(iv) for iv in winrate_masked.columns], rotation=45, ha="right")
ax.set_yticklabels([fmt_interval(iv) for iv in winrate_masked.index])

ax.set_xlabel("ncvd (quantile bins)")
ax.set_ylabel("ret_over_ncvd (quantile bins)")
ax.set_title(f"Conditional Win-rate Heatmap: P(hz_ret>0 | ncvd_bin, ret_over_ncvd_bin)\n"
             f"(min count per cell = {MIN_COUNT_PER_CELL}, rows={len(df)})")

# 각 칸에 "승률% / n" 텍스트 찍기
wr_vals = winrate.to_numpy()
ct_vals = counts.to_numpy()

for iy in range(winrate_masked.shape[0]):
    for ix in range(winrate_masked.shape[1]):
        n = ct_vals[iy, ix] if not np.isnan(ct_vals[iy, ix]) else 0
        wr = wr_vals[iy, ix]
        if np.isnan(winrate_masked.to_numpy()[iy, ix]):
            # 표본 부족이면 n만 작게 표시하거나 아예 스킵
            continue
        ax.text(ix, iy, f"{wr*100:.1f}%\n(n={int(n)})", ha="center", va="center", fontsize=8)

plt.tight_layout()
plt.show()
