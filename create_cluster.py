import json
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. パスの設定（app.py と同じフォルダに保存）
current_dir = os.path.dirname(os.path.abspath(__file__))
subset_path = os.path.join(current_dir, "clustering_subset.csv")
output_json = os.path.join(current_dir, "cluster0_keys.json")

print("1. クラスタリング用データを読み込み中...")
if not os.path.exists(subset_path):
  # フルパス指定のフォールバック
  subset_path = (
      r"C:\Users\fujiy\semi2026\data-analysis\application\clustering_subset.csv"
  )

df = pd.read_csv(subset_path)
print(f"データ読み込み完了: {len(df):,} 行")

# 2. クラスタリング変数の前処理
feature_cols = [
    "CBAA_33",
    "CBAA_34",
    "CCAB_2",
    "CEAA_3",
    "CEAA_4",
    "CEAD_3",
    "CBAA_1",
    "CBAA_4",
    "CBAA_10",
    "CBAA_29",
    "CBAA_30",
    "CGAA_10",
    "CGAA_11",
    "CGAA_12",
    "CGAA_1",
    "CGAA_2",
    "CGAA_6",
    "CGAA_7",
    "CGAA_8",
]

X = df[[c for c in feature_cols if c in df.columns]].copy()
for col in X.columns:
  X[col] = pd.to_numeric(X[col], errors="coerce").fillna(X[col].median())

print("2. k-means クラスタリングを実行中...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# 3. クラスター0のKEYを「文字列」として抽出
# ※KEY列を確実に文字列型に揃えることで、app.pyとの照合不一致を防止
df["KEY"] = df["KEY"].astype(str).str.strip()
c0_keys = df[df["Cluster"] == 0]["KEY"].tolist()

# 4. JSONファイルとして書き出し
with open(output_json, "w", encoding="utf-8") as f:
  json.dump(c0_keys, f, ensure_ascii=False)

print("\n" + "=" * 60)
print(f"🎉 成功！ cluster0_keys.json を生成しました。")
print(f"保存先: {output_json}")
print(f"クラスター0 人数: {len(c0_keys):,} 人 / 全体: {len(df):,} 人")
print(f"サンプルID: {c0_keys[:3]}")
print("=" * 60)