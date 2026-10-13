# estimate.py
import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "question_master.json")
PARQUET_PATH = os.path.join(BASE_DIR, "app_data.parquet")
CLUSTER_CSV = os.path.join(BASE_DIR, "starbucks_u30_users_with_cluster.csv")

# 1. マスターとクラスタ結果の読み込み
with open(JSON_PATH, "r", encoding="utf-8") as f:
  master = json.load(f)
questions = master["questions"]

if not os.path.exists(CLUSTER_CSV):
  print(f"エラー: {CLUSTER_CSV} が見つかりません。")
  sys.exit(1)

df_clusters = pd.read_csv(CLUSTER_CSV)
df_clusters["KEY"] = df_clusters["KEY"].astype(str)

# タイプ4を1、それ以外を0とする（タイプ4 = Cluster値が3）
TYPE4_CLUSTER_ID = 3
df_clusters["is_type4"] = (
    df_clusters["Cluster"] == TYPE4_CLUSTER_ID
).astype(int)
n_train_type4 = df_clusters["is_type4"].sum()
print(
    f"学習データ中のタイプ4人数: {n_train_type4} 人 / {len(df_clusters)} 人"
    f" ({(n_train_type4/len(df_clusters))*100:.1f}%)"
)

# 2. クラスタリングに使った生活意識設問のリストアップ
psycho_qs = []
for q_code, q_info in questions.items():
  cat = q_info.get("category", "")
  q_type = str(q_info.get("type", "")).upper()
  if q_type == "F" or q_code in ["CGAN_1", "AAF2"]:
    continue
  if any(keyword in cat for keyword in ["生活", "意識", "価値観", "消費", "態度"]):
    psycho_qs.append(q_code)

# 3. 全母集団データの読み込み
cols_to_load = list(set(["KEY", "CGAN_1", "AAF2"] + psycho_qs))
print("全母集団データを読み込み中...")
df_all = pd.read_parquet(PARQUET_PATH, columns=cols_to_load)
df_all["KEY"] = df_all["KEY"].astype(str)
n_total_pop = len(df_all)
print(f"全母集団サンプル数: {n_total_pop:,} 人")

# 4. 高速前処理
print("特徴量データを処理中...", flush=True)
feature_dict = {}

for q in psycho_qs:
  q_info = questions[q]
  q_type = str(q_info.get("type", "S")).upper()
  choices = q_info.get("choices", {})
  col_data = df_all[q]

  if q_type == "NL":
    feature_dict[q] = pd.to_numeric(col_data, errors="coerce").fillna(0).values
  elif q_type == "ML":
    dummies = col_data.astype(str).str.get_dummies(sep=",")
    for d_col in dummies.columns:
      feature_dict[f"{q}_opt{d_col}"] = dummies[d_col].values
  elif q_type in ["SL", "S"] and len(choices) in [4, 5]:
    first_choice = str(choices.get("1", ""))
    is_pos = any(
        w in first_choice for w in ["あてはまる", "そう思う", "非常に", "とても"]
    )
    s = pd.to_numeric(col_data, errors="coerce").fillna(0).values
    if is_pos:
      max_val = len(choices)
      s = np.where(s > 0, (max_val + 1) - s, 0)
    feature_dict[q] = s
  else:
    feature_dict[q] = pd.to_numeric(col_data, errors="coerce").fillna(0).values

X_all = pd.DataFrame(feature_dict, index=df_all.index)

# 5. 学習データの抽出とモデル学習
print("タイプ4推定モデルを学習中...", flush=True)
train_mask = df_all["KEY"].isin(df_clusters["KEY"])
train_sub = df_all.loc[train_mask, ["KEY"]].merge(
    df_clusters[["KEY", "is_type4"]], on="KEY", how="left"
)
y_train = train_sub["is_type4"].values
X_train = X_all.loc[train_mask]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_all_scaled = scaler.transform(X_all)

clf = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
clf.fit(X_train_scaled, y_train)

# 6. 全母集団の所属確率予測
print("全サンプルのタイプ4確率を推計中...", flush=True)
df_all["type4_prob"] = clf.predict_proba(X_all_scaled)[:, 1]

# 属性フラグ
age_num = pd.to_numeric(df_all["AAF2"], errors="coerce")
df_all["is_u30"] = age_num <= 30
df_all["is_starbucks_user"] = df_all["CGAN_1"].astype(str).isin(
    ["1", "2", "3", "4", "5"]
)

# 7. 結果テーブルの計算と表示
u30_total = df_all["is_u30"].sum()
print("\n" + "=" * 85)
print("📊 【タイプ4】市場規模推計シミュレーション（スタバ利用条件なし）")
print("=" * 85)
print(f"・調査全体の母集団: {n_total_pop:,} 人")
print(f"・うち 30歳以下全体: {u30_total:,} 人")
print("-" * 85)
print(
    f"{'判定パラメータ(確率閾値)':<22} | {'30歳以下での人数(構成比)':<20} |"
    f" {'全世代での人数(構成比)':<20} | {'スタバ高頻度層の割合'}"
)
print("-" * 85)

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

for th in thresholds:
  is_pred = df_all["type4_prob"] >= th

  # 30歳以下
  u30_cnt = (is_pred & df_all["is_u30"]).sum()
  u30_pct = (u30_cnt / u30_total) * 100 if u30_total > 0 else 0

  # 全世代
  all_cnt = is_pred.sum()
  all_pct = (all_cnt / n_total_pop) * 100

  # タイプ4判定者におけるスタバ高頻度率（30歳以下ベース）
  sb_users_in_type4 = (
      is_pred & df_all["is_u30"] & df_all["is_starbucks_user"]
  ).sum()
  sb_ratio = (sb_users_in_type4 / u30_cnt) * 100 if u30_cnt > 0 else 0

  label = f"閾値 {th:.1f} 以上"
  if th == 0.5:
    label += " (標準判定)"
  elif th == 0.7:
    label += " (コア層)"
  elif th == 0.3:
    label += " (広義・潜在層)"

  print(
      f"{label:<22} | {u30_cnt:>6,d}人 ({u30_pct:>5.1f}%)     |"
      f" {all_cnt:>6,d}人 ({all_pct:>5.1f}%)     | {sb_ratio:>5.1f}%"
  )

print("=" * 85)
print("📌 解釈のポイント:")
print(
    "1. 30歳以下における出現率: 世の中の若年層の中でタイプ4の価値観を持つ人がどれくらいいるか"
)
print(
    "2. スタバ高頻度層の割合: そのうち既にスタバによく行っている人の割合（＝残りは未開拓の伸びしろ市場）"
)

# 結果CSVも保存（ID, 年齢, スタバ利用, タイプ4確率）
output_csv = os.path.join(BASE_DIR, "type4_market_estimates.csv")
df_all[["KEY", "AAF2", "CGAN_1", "is_u30", "is_starbucks_user", "type4_prob"]].to_csv(
    output_csv, index=False, encoding="utf-8-sig"
)
print(f"\n💾 詳細データを保存しました: {output_csv}")