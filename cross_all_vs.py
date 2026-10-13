# crosstab_type4_vs_all.py
import json
import os
import re
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "question_master.json")
PARQUET_PATH = os.path.join(BASE_DIR, "app_data.parquet")
ESTIMATE_CSV = os.path.join(BASE_DIR, "type4_market_estimates.csv")

# 1. 設問マスターと推計データの読み込み
with open(JSON_PATH, "r", encoding="utf-8") as f:
  master = json.load(f)
questions = master["questions"]

if not os.path.exists(ESTIMATE_CSV):
  print(
      f"エラー: {ESTIMATE_CSV} が見つかりません。先に estimate.py を実行してください。"
  )
  exit()

df_est = pd.read_csv(ESTIMATE_CSV)
df_est["KEY"] = df_est["KEY"].astype(str)

# タイプ4潜在層（30歳以下 × 確率0.3以上）のキーを特定
type4_keys = set(
    df_est.loc[
        (df_est["is_u30"] == True) & (df_est["type4_prob"] >= 0.3), "KEY"
    ]
)
print(f"🎯 タイプ4潜在層: {len(type4_keys):,} 人")

# ----------------------------------------------------
# 2. 集計したいテーマ・大分類のキーワードを指定
# ----------------------------------------------------
# ここに気になるキーワードを並べてください（空にすると全設問対象）
SEARCH_KEYWORDS = ["SNS", "情報", "メディア", "休日", "趣味", "飲料", "消費"]


def clean_text(text):
  return re.sub(r"[※■★◆\s]+", " ", str(text)).strip()


# 対象設問の選定
target_qs = []
for q_code, q_info in questions.items():
  cat = q_info.get("category", "")
  title = q_info.get("title", "")
  q_type = str(q_info.get("type", "")).upper()

  # 自由記述や属性ID等は除外
  if q_type == "F" or q_code in ["CGAN_1", "AAF2"]:
    continue

  if not SEARCH_KEYWORDS or any(
      kw in cat or kw in title for kw in SEARCH_KEYWORDS
  ):
    target_qs.append(q_code)

print(
    f"集計対象設問数: {len(target_qs)} 問（検索キーワード:"
    f" {SEARCH_KEYWORDS}）"
)

# 3. 必要な列をParquetから読み込み（30歳以下に限定）
cols_to_load = list(set(["KEY", "AAF2"] + target_qs))
print("Parquetからデータを読み込み中...", flush=True)
df_all = pd.read_parquet(PARQUET_PATH, columns=cols_to_load)
df_all["KEY"] = df_all["KEY"].astype(str)

# 30歳以下全体にフィルタ
age_s = pd.to_numeric(df_all["AAF2"], errors="coerce")
df_u30 = df_all[age_s <= 30].copy()
n_u30_total = len(df_u30)
print(f"30歳以下全体母集団: {n_u30_total:,} 人")

# タイプ4フラグを付与
df_u30["is_type4"] = df_u30["KEY"].isin(type4_keys)
group_t4 = df_u30[df_u30["is_type4"]]
n_t4 = len(group_t4)

# ----------------------------------------------------
# 4. クロス集計処理
# ----------------------------------------------------
records = []

for q_code in target_qs:
  q_info = questions[q_code]
  q_title = q_info.get("title", q_code)
  q_type = str(q_info.get("type", "S")).upper()
  choices = q_info.get("choices", {})
  cat = clean_text(q_info.get("category", ""))

  col_t4 = group_t4[q_code]
  col_all = df_u30[q_code]

  # Type ML: カンマ区切りの複数回答
  if q_type == "ML":
    dum_t4 = col_t4.astype(str).str.get_dummies(sep=",")
    dum_all = col_all.astype(str).str.get_dummies(sep=",")

    opts = sorted(list(set(dum_t4.columns).union(set(dum_all.columns))))
    for opt in opts:
      opt_name = choices.get(opt, f"選択肢{opt}")
      cnt_t4 = dum_t4[opt].sum() if opt in dum_t4 else 0
      cnt_all = dum_all[opt].sum() if opt in dum_all else 0

      pct_t4 = (cnt_t4 / n_t4) * 100 if n_t4 > 0 else 0
      pct_all = (cnt_all / n_u30_total) * 100 if n_u30_total > 0 else 0
      diff = pct_t4 - pct_all

      records.append({
          "大分類": cat,
          "設問コード": q_code,
          "設問タイトル": q_title,
          "回答・選択肢": opt_name,
          "指標": "選択率(%)",
          "タイプ4潜在層 (n=325)": round(pct_t4, 1),
          "U30全体 (n=3,689)": round(pct_all, 1),
          "差異 (%pt)": round(diff, 1),
      })

  # Type N / NL: 数値項目
  elif q_type in ["N", "NL"]:
    s_t4 = pd.to_numeric(col_t4, errors="coerce")
    s_all = pd.to_numeric(col_all, errors="coerce")
    if q_type == "NL":
      s_t4 = s_t4.fillna(0)
      s_all = s_all.fillna(0)

    mean_t4 = s_t4.mean()
    mean_all = s_all.mean()
    diff = mean_t4 - mean_all

    records.append({
        "大分類": cat,
        "設問コード": q_code,
        "設問タイトル": q_title,
        "回答・選択肢": "平均値",
        "指標": "平均値",
        "タイプ4潜在層 (n=325)": round(mean_t4, 2),
        "U30全体 (n=3,689)": round(mean_all, 2),
        "差異 (%pt)": round(diff, 2),
    })

  # Type S / SL / M: 単一回答・マトリクス等
  else:
    s_t4 = col_t4.astype(str)
    s_all = col_all.astype(str)

    unique_vals = [
        v for v in choices.keys() if v in s_all.unique() and v != "nan"
    ]
    if not unique_vals:
      unique_vals = [
          v for v in s_all.unique() if v not in ["nan", "None", "", "0"]
      ]

    for val in unique_vals:
      opt_name = choices.get(val, f"コード{val}")
      cnt_t4 = (s_t4 == val).sum()
      cnt_all = (s_all == val).sum()

      pct_t4 = (cnt_t4 / n_t4) * 100 if n_t4 > 0 else 0
      pct_all = (cnt_all / n_u30_total) * 100 if n_u30_total > 0 else 0
      diff = pct_t4 - pct_all

      records.append({
          "大分類": cat,
          "設問コード": q_code,
          "設問タイトル": q_title,
          "回答・選択肢": opt_name,
          "指標": "回答比率(%)",
          "タイプ4潜在層 (n=325)": round(pct_t4, 1),
          "U30全体 (n=3,689)": round(pct_all, 1),
          "差異 (%pt)": round(diff, 1),
      })

# ----------------------------------------------------
# 5. 並び替えと保存
# ----------------------------------------------------
df_res = pd.DataFrame(records)
# タイプ4が同世代平均より突出している順（差異が大きい順）にソート
df_res.sort_values(by="差異 (%pt)", ascending=False, inplace=True)

output_csv = os.path.join(BASE_DIR, "type4_vs_u30all_comparison.csv")
try:
  df_res.to_csv(output_csv, index=False, encoding="utf-8-sig")
  print("\n" + "=" * 70)
  print(f"✅ 集計完了！比較表をCSV保存しました: {output_csv}")
  print("=" * 70)
except PermissionError:
  output_csv_alt = os.path.join(
      BASE_DIR, "type4_vs_u30all_comparison_new.csv"
  )
  df_res.to_csv(output_csv_alt, index=False, encoding="utf-8-sig")
  print(f"⚠️ ファイルがロックされていたため別名で保存: {output_csv_alt}")

# ターミナルに「タイプ4がU30全体より顕著に高い項目 TOP 15」を表示
print("\n🔥 【タイプ4が同世代（30歳以下全体）より突出して高い項目 TOP 15】")
preview_cols = [
    "大分類",
    "設問タイトル",
    "回答・選択肢",
    "タイプ4潜在層 (n=325)",
    "U30全体 (n=3,689)",
    "差異 (%pt)",
]
print(df_res[preview_cols].head(15).to_string(index=False))