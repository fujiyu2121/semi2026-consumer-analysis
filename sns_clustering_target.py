# sns_clustering_target.py (NumPy単独・低メモリ版)
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_PATH = os.path.join(BASE_DIR, "app_data.parquet")
OUTPUT_CSV = os.path.join(BASE_DIR, "u30_sns_target_clusters.csv")

cols_to_load = [
    "KEY",
    "AAF2",
    "HDAB_4",
    "HDAB_14",
    "HEAB_23",
    "CFAJ_5",
    "HDAK_4",
    "CCAB_21",
    "CEAE",
    "CBAB_1",
    "CCAB_22",
    "IBAF_7",
]

print("Parquetからデータを読み込み中...", flush=True)
df = pd.read_parquet(PARQUET_PATH, columns=cols_to_load)
df["KEY"] = df["KEY"].astype(str)

age_s = pd.to_numeric(df["AAF2"], errors="coerce")
df_u30 = df[age_s <= 30].copy().reset_index(drop=True)
n_u30 = len(df_u30)
print(f"分析対象（30歳以下）: {n_u30:,} 人")

features = pd.DataFrame(index=df_u30.index)


def invert_scale(series, max_val):
  s = pd.to_numeric(series, errors="coerce").fillna(0)
  return np.where(s > 0, (max_val + 1) - s, 0)


# ① SNS視聴頻度
features["f_ig_freq"] = invert_scale(df_u30["HDAB_4"], 6)
features["f_tiktok_freq"] = invert_scale(df_u30["HDAB_14"], 6)
features["f_tiktok_video"] = invert_scale(df_u30["HEAB_23"], 6)

# ② タイムライン反応・衝動買い
features["f_ad_click"] = (
    pd.to_numeric(df_u30["CFAJ_5"], errors="coerce") == 1
).astype(int)
features["f_buy_ig_ad"] = (
    df_u30["HDAK_4"].astype(str).str.contains(r"\b10\b").astype(int)
)
features["f_impulse_buy"] = invert_scale(df_u30["CCAB_21"], 4)

# ③ 美容・外見への自己投資
features["f_interest_beauty"] = (
    df_u30["CEAE"].astype(str).str.contains(r"\b14\b").astype(int)
)
features["f_spend_appearance"] = (
    df_u30["CBAB_1"].astype(str).str.contains(r"\b9\b").astype(int)
)
features["f_spend_fashion"] = invert_scale(df_u30["CCAB_22"], 4)

# ④ コンビニ利用頻度
features["f_cstore_freq"] = invert_scale(df_u30["IBAF_7"], 5)

# 標準化（NumPy実装）
X = features.values.astype(np.float32)
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
std[std == 0] = 1.0
X_scaled = (X - mean) / std


# 軽量 k-means（NumPy実装）
def simple_kmeans(X, n_clusters=4, max_iter=100, seed=42):
  np.random.seed(seed)
  idx = np.random.choice(len(X), n_clusters, replace=False)
  centers = X[idx].copy()

  for _ in range(max_iter):
    # ユークリッド距離計算
    dists = np.linalg.norm(X[:, np.newaxis, :] - centers, axis=2)
    labels = np.argmin(dists, axis=1)

    new_centers = np.array(
        [
            X[labels == k].mean(axis=0) if np.sum(labels == k) > 0 else centers[k]
            for k in range(n_clusters)
        ]
    )
    if np.allclose(centers, new_centers, atol=1e-4):
      break
    centers = new_centers
  return labels


print("クラスタリング計算中...", flush=True)
n_clusters = 4
df_u30["Cluster"] = simple_kmeans(X_scaled, n_clusters=n_clusters)

# 結果集計
print("\n" + "=" * 75)
print("📊 【クラスタ別 特徴量平均値一覧（Z-Score）】")
print("=" * 75)

cluster_summary = (
    pd.DataFrame(X_scaled, columns=features.columns)
    .groupby(df_u30["Cluster"])
    .mean()
)
cluster_sizes = df_u30["Cluster"].value_counts().sort_index()

for c_id in range(n_clusters):
  cnt = cluster_sizes[c_id]
  pct = (cnt / n_u30) * 100
  print(f"\n▼ 【クラスタ {c_id + 1}】 規模: {cnt:,} 人 ({pct:.1f}%)")
  top_features = cluster_summary.loc[c_id].sort_values(ascending=False)
  for col, val in top_features.items():
    bar = "+" * int(val * 5) if val > 0 else "-" * int(abs(val) * 5)
    print(f"  ・{col:<18}: {val:>+5.2f}  {bar}")

df_u30[["KEY", "AAF2", "Cluster"]].to_csv(
    OUTPUT_CSV, index=False, encoding="utf-8-sig"
)
print(f"\n💾 保存完了: {OUTPUT_CSV}")