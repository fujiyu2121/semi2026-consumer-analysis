# cluster_starbucks.py
import json
import os
import re
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# スクリプト自身のディレクトリを取得
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "question_master.json")
PARQUET_PATH = os.path.join(BASE_DIR, "app_data.parquet")

# 1. 設問マスターの読み込み
with open(JSON_PATH, "r", encoding="utf-8") as f:
  master = json.load(f)
questions = master["questions"]

TARGET_Q = "CGAN_1"
TARGET_VALUES = ["1", "2", "3", "4", "5"]  # スタバ月2〜3日以上の利用層
MAX_AGE = 30  # 30歳以下

# 2. クラスタリングに使う「生活意識・価値観」設問を抽出
psycho_qs = []
for q_code, q_info in questions.items():
  cat = q_info.get("category", "")
  q_type = str(q_info.get("type", "")).upper()

  if q_type == "F" or q_code in [TARGET_Q, "AAF2"]:
    continue

  if any(keyword in cat for keyword in ["生活", "意識", "価値観", "消費", "態度"]):
    psycho_qs.append(q_code)

print(f"抽出された分析対象設問数: {len(psycho_qs)} 問")

# 3. 必要な列をピンポイント読み込み
cols_to_load = list(set(["KEY", TARGET_Q, "AAF2"] + psycho_qs))
df = pd.read_parquet(PARQUET_PATH, columns=cols_to_load)

# 条件フィルタリング（スタバ月2〜3日以上 ＆ 30歳以下）
age_series = pd.to_numeric(df["AAF2"], errors="coerce")
cond_starbucks = df[TARGET_Q].astype(str).isin(TARGET_VALUES)
cond_age = age_series <= MAX_AGE

df_target = df[cond_starbucks & cond_age].copy()
n_target = len(df_target)
print(
    f"分析対象（スタバ月2〜3日以上 ＆ 30歳以下）のサンプル数: {n_target:,} 人"
)

if n_target == 0:
  print("対象者が0人です。")
  exit()

# 4. 各設問Typeに応じた前処理（スコア反転処理を追加）
feature_dfs = []
reversed_cols = set()

for q in psycho_qs:
  q_info = questions[q]
  q_type = str(q_info.get("type", "S")).upper()
  col_data = df_target[q]
  choices = q_info.get("choices", {})

  # Type NL: 数値内訳（未回答は0埋め）
  if q_type == "NL":
    s = pd.to_numeric(col_data, errors="coerce").fillna(0)
    feature_dfs.append(pd.DataFrame({q: s}, index=df_target.index))

  # Type ML: カンマ区切り複数回答（選択肢ごとにダミー化）
  elif q_type == "ML":
    dummies = col_data.astype(str).str.get_dummies(sep=",")
    dummies.columns = [f"{q}_opt{c}" for c in dummies.columns]
    feature_dfs.append(dummies)

  # Type SL / S: 4段階・5段階リッカート尺度の反転処理
  elif q_type in ["SL", "S"] and len(choices) in [4, 5]:
    # 選択肢'1'が「あてはまる」系か判定
    first_choice = str(choices.get("1", ""))
    is_positive_first = any(
        w in first_choice for w in ["あてはまる", "そう思う", "非常に", "とても"]
    )

    s = pd.to_numeric(col_data, errors="coerce").fillna(0)
    if is_positive_first:
      # 1(高)〜4(低) なら (4 + 1) - s で 4(高)〜1(低) に反転
      max_val = len(choices)
      s = np.where(s > 0, (max_val + 1) - s, 0)
      reversed_cols.add(q)
    feature_dfs.append(pd.DataFrame({q: s}, index=df_target.index))

  else:
    s = pd.to_numeric(col_data, errors="coerce").fillna(0)
    feature_dfs.append(pd.DataFrame({q: s}, index=df_target.index))

X_raw = pd.concat(feature_dfs, axis=1)
X_clean = X_raw.loc[:, X_raw.std() > 0].copy()
print(f"機械学習に投入する特徴量数: {X_clean.shape[1]} 変数")

# 5. 標準化 ＆ PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

pca = PCA(n_components=min(30, X_scaled.shape[1]), random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 6. k-means クラスタリング（4タイプ）
K = 4
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
df_target["Cluster"] = kmeans.fit_predict(X_pca)

# 7. 各クラスタの特徴量分析
cluster_means = pd.DataFrame(X_scaled, columns=X_clean.columns).groupby(
    df_target["Cluster"].values
).mean()


# カテゴリ名のクレンジング用関数
def clean_category(cat_str):
  return re.sub(r"[※■★◆\s]+", " ", cat_str).strip()


# 8. ファイル保存
txt_output_path = os.path.join(BASE_DIR, "starbucks_u30_cluster_report.txt")
csv_output_path = os.path.join(BASE_DIR, "starbucks_u30_cluster_features.csv")

# CSVデータの整形
records = []
for c in range(K):
  for col_name, z_score in cluster_means.loc[c].items():
    base_q = col_name.split("_opt")[0]
    q_info = questions.get(base_q, {})
    q_title = q_info.get("title", base_q)
    raw_cat = q_info.get("category", "")
    cat_clean = clean_category(raw_cat)

    if "_opt" in col_name:
      opt_id = col_name.split("_opt")[1]
      opt_label = q_info.get("choices", {}).get(opt_id, f"選択肢{opt_id}")
      item_title = f"{q_title} 【{opt_label}】"
      trend = "利用/選択割合が高い" if z_score > 0 else "利用/選択割合が低い"
    else:
      item_title = q_title
      if base_q in reversed_cols:
        trend = (
            "強く共感・あてはまる" if z_score > 0 else "共感しない・あてはまらない"
        )
      else:
        trend = "スコアが高い" if z_score > 0 else "スコアが低い"

    records.append({
        "クラスタ": f"タイプ {c+1}",
        "設問コード": base_q,
        "大分類": cat_clean,
        "項目・意識（設問タイトル）": item_title,
        "傾向・解釈": trend,
        "偏りスコア(Z-Score)": round(z_score, 3),
    })

df_features = pd.DataFrame(records)
df_features.sort_values(
    ["クラスタ", "偏りスコア(Z-Score)"], ascending=[True, False], inplace=True
)
df_features.to_csv(csv_output_path, index=False, encoding="utf-8-sig")

# テキスト要約レポート保存（TOP 10）
with open(txt_output_path, "w", encoding="utf-8") as f:
  f.write("=" * 65 + "\n")
  f.write("🥤 【30歳以下 × スタバ高頻度層】生活者タイプ別プロファイリング結果\n")
  f.write("=" * 65 + "\n")
  f.write(f"対象者総数: {n_target:,} 人（スタバ月2〜3日以上 ＆ 30歳以下）\n")

  for c in range(K):
    c_size = (df_target["Cluster"] == c).sum()
    c_ratio = (c_size / n_target) * 100
    avg_age = pd.to_numeric(
        df_target.loc[df_target["Cluster"] == c, "AAF2"], errors="coerce"
    ).mean()

    f.write(
        f"\n【タイプ {c+1}】 規模: {c_size:,}人 ({c_ratio:.1f}%) ｜ 平均年齢:"
        f" {avg_age:.1f}歳\n"
    )
    f.write("  --- 全体平均と比べて特に共感・傾向が強い意識 TOP 10 ---\n")

    sub_df = df_features[
        (df_features["クラスタ"] == f"タイプ {c+1}")
        & (df_features["偏りスコア(Z-Score)"] > 0)
    ].head(10)
    for _, row in sub_df.iterrows():
      f.write(
          f"   ★ [{row['大分類']}] {row['項目・意識（設問タイトル）']} (スコア:"
          f" +{row['偏りスコア(Z-Score)']:.2f})\n"
      )

# 回答者IDとクラスタの対応表保存
df_target[["KEY", "AAF2", "Cluster"]].to_csv(
    os.path.join(BASE_DIR, "starbucks_u30_users_with_cluster.csv"),
    index=False,
    encoding="utf-8-sig",
)

print("\n" + "=" * 65)
print("プロファイリング完了！ファイルを生成しました。")
print(f"  ・要約レポート: {txt_output_path}")
print(f"  ・特徴量一覧CSV: {csv_output_path}")
print("=" * 65)