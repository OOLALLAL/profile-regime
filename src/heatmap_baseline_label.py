import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = r"D:\data\profile-regime\experiments\binance\futures_um\symbol=BTCUSDT\event_features\event_horizon_features.parquet"

USE_BAND_FILTER = True
BAND_FILTER = {
    "tf": ["15min"],
    "w": [96],
    "m": [2, 4],
    "g": [0, 1],
    "h": [16],
}
QX = 5
QY = 5

EV_FEATURE_IDX = "ret_over_ncvd"
EV_FEATURE_COL = "ncvd"
HZ_FEATURE = "hz_max_dn"
MIN_HZ_FEATURE = -0.007
MIN_COUNT_PER_CELL = 0

df = pd.read_parquet(DATA_PATH)

features = [EV_FEATURE_IDX, EV_FEATURE_IDX, HZ_FEATURE]
missing = [c for c in features if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

if USE_BAND_FILTER:
    mask = (
        df["tf"].isin(BAND_FILTER["tf"])
        & df["w"].isin(BAND_FILTER["w"])
        & df["m"].isin(BAND_FILTER["m"])
        & df["g"].isin(BAND_FILTER["g"])
        & df["h"].isin(BAND_FILTER["h"])
    )
    df = df.loc[mask].copy()

df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features).copy()



df["win"] = (df[HZ_FEATURE] > MIN_HZ_FEATURE).astype(int)

print("Rows after filter: ", len(df))

df["bin_x"] = pd.qcut(df[EV_FEATURE_COL], q=QX, duplicates="drop")
df["bin_y"] = pd.qcut(df[EV_FEATURE_IDX], q=QY, duplicates="drop")

x_bin = df["bin_x"].cat.categories
y_bin = df["bin_y"].cat.categories
print(f"Actual bins: {len(x_bin)} X {len(y_bin)}")

winrates = df.pivot_table(index="bin_y", columns="bin_x", values="win", aggfunc="mean", observed=False)
counts   = df.pivot_table(index="bin_y", columns="bin_x", values="win", aggfunc="size", observed=False)

winrate_masked = winrates.where(counts >= MIN_COUNT_PER_CELL)

# ===== plotting =====
fig ,ax = plt.subplots(figsize=(12, 9))

im = ax.imshow(winrate_masked.to_numpy(), origin="lower", aspect="auto")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label(f"P({HZ_FEATURE} > 0)")


def fmt_interval(iv):
    # iv: pandas Interval
    return f"({iv.left:.4f}, {iv.right:.4f}]"

ax.set_xticks(np.arange(winrate_masked.shape[1]))
ax.set_yticks(np.arange(winrate_masked.shape[0]))
ax.set_xticklabels([fmt_interval(iv) for iv in winrate_masked.columns], rotation=45, ha="right")
ax.set_yticklabels([fmt_interval(iv) for iv in winrate_masked.index])

ax.set_xlabel(f"{EV_FEATURE_COL} (quantile bins)")
ax.set_ylabel(f"{EV_FEATURE_IDX} (quantile bins)")
ax.set_title(f"Conditional Win-rate Heatmap: P({HZ_FEATURE}>{MIN_HZ_FEATURE*100:.2f}% | {EV_FEATURE_COL}_bin, {EV_FEATURE_IDX}_bin)\n"
             f"(min count per cell = {MIN_COUNT_PER_CELL}, rows={len(df)})")

wr_vals = winrates.to_numpy()
ct_vals = counts.to_numpy()

for idx in range(winrate_masked.shape[0]):
    for col in range(winrate_masked.shape[1]):
        ct = ct_vals[idx, col] if not np.isnan(ct_vals[idx, col]) else 0
        wr = wr_vals[idx, col]
        if np.isnan(winrate_masked.to_numpy()[idx, col]):
            continue
        ax.text(col, idx, f"{wr*100:.1f}%\n(n={ct})", ha="center", va="center", fontsize=8)

plt.tight_layout()
plt.show()