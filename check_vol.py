import pandas as pd
import numpy as np

df = pd.read_parquet("data/features/BTC_features.parquet")
mid = df["mid"].values
idx = df.index

print(f"Avg BTC mid price: ${mid.mean():,.0f}\n")
print(f"{'Horizon':<10} {'std':>8} {'p10':>8} {'p25':>8} {'p50':>8} {'p75':>8} {'p90':>8}  meaning")
print("-" * 85)

for sec in [1, 2, 5, 10, 20, 30, 60]:
    t_fwd   = idx + pd.Timedelta(seconds=sec)
    fwd_pos = idx.searchsorted(t_fwd)
    fwd_pos = np.clip(fwd_pos, 0, len(idx) - 1)
    moves   = mid[fwd_pos] - mid
    t_diff  = (idx[fwd_pos] - idx).total_seconds()
    valid   = np.abs(t_diff - sec) <= 2
    m       = moves[valid]
    print(f"{str(sec)+'s':<10} {m.std():>8.3f} {np.percentile(m,10):>8.3f} "
          f"{np.percentile(m,25):>8.3f} {np.percentile(m,50):>8.3f} "
          f"{np.percentile(m,75):>8.3f} {np.percentile(m,90):>8.3f}"
          f"  <- 80% of {sec}s moves are within p10..p90")