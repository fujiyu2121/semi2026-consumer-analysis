import json
import os
import re
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "question_master.json")
PARQUET_PATH = os.path.join(BASE_DIR, "app_data.parquet")
CLUSTER_CSV = os.path.join(BASE_DIR, "starbucks_u30_users_with_cluster.csv")

# 1. 前提データの読み込み
with open(JSON_PATH, "r", encoding="utf-8") as f:
  master = json.load(f)
questions = master["questions"]

if not os.path.exists(CLUSTER_CSV):
  print(
      f"エラー: {CLUSTER_CSV} が見つかりません。先に"
      " cluster_starbucks.py を実行してください。"
  )
  exit()

df_clusters = pd.read_csv(CLUSTER_CSV)
n_users = len(df_clusters)

# クラスタ名マッピング
cluster_col = "Cluster"
cluster_names = {
    c: f"タイプ{c+1}" for c in sorted(df_clusters[cluster_col].unique())
}
df_clusters["Cluster_Label"] = df_clusters[cluster_col].map(cluster_names)

# 各タイプのサンプル数
sample_sizes = df_clusters["Cluster_Label"].value_counts().to_dict()

# ----------------------------------------------------
# 2. 集計したい大分類・テーマのキーワードを指定
# ----------------------------------------------------
# ここに調べたい言葉を自由に追加・変更できます
SEARCH_KEYWORDS = ["SNS", "情報", "メディア", "休日", "趣味", "飲料", "消費"]


def clean_text(text):
  return re.sub(r"[※■★◆\s]+", " ", str(text)).strip()


# 対象設問を抽出（自由記述Fと既にクラスタリングした設問等は除外）
target_qs = []
for q_code, q_info in questions.items():
  cat = q_info.get("category", "")
  q_type = str(q_info.get("type", "")).upper()
  if q_type == "F" or q_code in ["CGAN_1"]:
    continue

  if any(kw in cat or kw in q_info.get("title", "") for kw in SEARCH_KEYWORDS):
    target_qs.append(q_code)

print(
    f"抽出された分析対象設問: {len(target_qs)} 問（検索キーワード:"
    f" {SEARCH_KEYWORDS}）"
)

if not target_qs:
  print("該当する設問が見つかりませんでした。キーワードを変更してください。")
  exit()

# 3. 必要な列をParquetからオンデマンド読み込み
cols_to_load = list(set(["KEY"] + target_qs))
df_data = pd.read_parquet(PARQUET_PATH, columns=cols_to_load)

# KEY列の型を文字列型に統一して安全に結合
df_clusters["KEY"] = df_clusters["KEY"].astype(str)
df_data["KEY"] = df_data["KEY"].astype(str)

df_merged = pd.merge(df_clusters, df_data, on="KEY", how="inner")

# ----------------------------------------------------
# 4. タイプ別クロス集計の実行
# ----------------------------------------------------
records = []
all_types = [cluster_names[c] for c in sorted(cluster_names.keys())]

for q_code in target_qs:
  q_info = questions[q_code]
  q_title = q_info.get("title", q_code)
  q_type = str(q_info.get("type", "S")).upper()
  choices = q_info.get("choices", {})
  cat = clean_text(q_info.get("category", ""))

  col_data = df_merged[q_code]

  # --- Type ML: カンマ区切り複数回答 ---
  if q_type == "ML":
    dummies = col_data.astype(str).str.get_dummies(sep=",")
    for opt_id in dummies.columns:
      opt_label = choices.get(opt_id, f"選択肢{opt_id}")
      row_dict = {
          "大分類": cat,
          "設問コード": q_code,
          "設問タイトル": q_title,
          "回答・選択肢": opt_label,
          "指標": "選択率(%)",
      }
      # 各タイプの選択率
      for t in all_types:
        subset = dummies.loc[df_merged["Cluster_Label"] == t, opt_id]
        row_dict[t] = round(subset.mean() * 100, 1) if len(subset) > 0 else 0.0
      row_dict["全体平均"] = round(dummies[opt_id].mean() * 100, 1)
      records.append(row_dict)

  # --- Type N / NL: 数値回答 ---
  elif q_type in ["N", "NL"]:
    s_num = pd.to_numeric(col_data, errors="coerce")
    if q_type == "NL":
      s_num = s_num.fillna(0)
    row_dict = {
        "大分類": cat,
        "設問コード": q_code,
        "設問タイトル": q_title,
        "回答・選択肢": "平均値",
        "指標": "平均値",
    }
    for t in all_types:
      subset = s_num[df_merged["Cluster_Label"] == t]
      row_dict[t] = round(subset.mean(), 2) if len(subset) > 0 else 0.0
    row_dict["全体平均"] = round(s_num.mean(), 2)
    records.append(row_dict)

  # --- Type S / SL / M: 単一回答・マトリクス等 ---
  else:
    s_clean = col_data.astype(str)
    unique_vals = [
        v for v in choices.keys() if v in s_clean.unique() and v != "nan"
    ]
    if not unique_vals:
      unique_vals = [v for v in s_clean.unique() if v not in ["nan", "None", ""]]

    for val in unique_vals:
      opt_label = choices.get(val, f"コード{val}")
      row_dict = {
          "大分類": cat,
          "設問コード": q_code,
          "設問タイトル": q_title,
          "回答・選択肢": opt_label,
          "指標": "回答比率(%)",
      }
      for t in all_types:
        total_in_cluster = (df_merged["Cluster_Label"] == t).sum()
        count = ((df_merged["Cluster_Label"] == t) & (s_clean == val)).sum()
        pct = (
            (count / total_in_cluster * 100) if total_in_cluster > 0 else 0.0
        )
        row_dict[t] = round(pct, 1)

      total_count = (s_clean == val).sum()
      row_dict["全体平均"] = round(total_count / len(df_merged) * 100, 1)
      records.append(row_dict)

# ----------------------------------------------------
# 5. 結果の保存と表示
# ----------------------------------------------------
df_result = pd.DataFrame(records)

# 差が顕著な項目（タイプ間の最大値 - 最小値）を計算して並び替え
type_cols = all_types
df_result["最大ギャップ(%pt)"] = (
    df_result[type_cols].max(axis=1) - df_result[type_cols].min(axis=1)
).round(1)
df_result.sort_values(by="最大ギャップ(%pt)", ascending=False, inplace=True)

output_csv = os.path.join(BASE_DIR, "clusters_crosstab_comparison.csv")
try:
  df_result.to_csv(output_csv, index=False, encoding="utf-8-sig")
  print("\n" + "=" * 65)
  print(f"✅ 集計完了！Excel用ファイルに出力しました: {output_csv}")
  print("=" * 65)
except PermissionError:
  output_csv_alt = os.path.join(
      BASE_DIR, "clusters_crosstab_comparison_new.csv"
  )
  df_result.to_csv(output_csv_alt, index=False, encoding="utf-8-sig")
  print(f"⚠️ ファイルがロックされていたため別名で保存: {output_csv_alt}")

# ターミナルに「タイプ間で最も差がついた項目 TOP 10」を表示
print("\n🔥 【タイプ間で差が大きかった行動・回答 TOP 10】")
preview_cols = [
    "大分類",
    "設問タイトル",
    "回答・選択肢",
] + all_types + ["全体平均", "最大ギャップ(%pt)"]
print(df_result[preview_cols].head(10).to_string(index=False))