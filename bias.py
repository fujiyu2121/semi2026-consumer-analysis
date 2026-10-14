# bias.py
import json
import os
import re
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_PATH = os.path.join(BASE_DIR, "app_data.parquet")
CLUSTER_CSV = os.path.join(BASE_DIR, "u30_sns_target_clusters.csv")
MASTER_JSON = os.path.join(BASE_DIR, "question_master.json")
OUTPUT_CSV = os.path.join(BASE_DIR, "cluster4_bias_ranking.csv")

print("1. 設問マスターとクラスタキーを読み込み中...", flush=True)

# 設問マスター
with open(MASTER_JSON, "r", encoding="utf-8") as f:
  master = json.load(f)
questions = master["questions"]

# クラスタ割り当てデータ
df_clusters = pd.read_csv(CLUSTER_CSV)
df_clusters["KEY"] = df_clusters["KEY"].astype(str)

# クラスタ4 (Cluster == 3) のKEY特定
c4_keys = set(df_clusters.loc[df_clusters["Cluster"] == 3, "KEY"])
n_c4 = len(c4_keys)
print(f"🎯 クラスタ4 対象人数: {n_c4:,} 人")

# まず KEY と 年齢(AAF2) だけを読み込んで対象インデックスとマスクを確定
df_base = pd.read_parquet(PARQUET_PATH, columns=["KEY", "AAF2"])
df_base["KEY"] = df_base["KEY"].astype(str)
age_s = pd.to_numeric(df_base["AAF2"], errors="coerce")

# 30歳以下の行マスク
u30_mask = age_s <= 30
u30_indices = df_base[u30_mask].index
n_u30 = len(u30_indices)
print(f"👥 30歳以下全体人数: {n_u30:,} 人")

# クラスタ4の行マスク（30歳以下の中での位置）
c4_mask_u30 = df_base.loc[u30_mask, "KEY"].isin(c4_keys).values

# 分析対象の設問リスト（自由記述等を除外）
target_qs = [
    q
    for q, info in questions.items()
    if info.get("type") not in ["F", "X"] and q not in ["KEY", "AAF2"]
]
print(
    f"2. 全 {len(target_qs)} 問を分割処理（メモリ節約）で集計します...",
    flush=True,
)


def clean_str(s):
  return re.sub(r"[\r\n\t]+", " ", str(s)).strip()


# 100問ずつチャンクに分けて処理
CHUNK_SIZE = 100
records = []

for i in range(0, len(target_qs), CHUNK_SIZE):
  chunk_qs = target_qs[i : i + CHUNK_SIZE]
  # 必要な列だけを都度ロードし、30歳以下の行だけ即座に抽出
  df_chunk = pd.read_parquet(PARQUET_PATH, columns=chunk_qs).loc[u30_indices]

  for q_code in chunk_qs:
    q_info = questions[q_code]
    cat = clean_str(q_info.get("category", ""))
    title = clean_str(q_info.get("title", q_code))
    q_type = str(q_info.get("type", "S")).upper()
    choices = q_info.get("choices", {})

    s_all = df_chunk[q_code]
    s_c4 = s_all[c4_mask_u30]

    # 複数回答 (ML)
    if q_type == "ML":
      dum_c4 = s_c4.astype(str).str.get_dummies(sep=",")
      dum_all = s_all.astype(str).str.get_dummies(sep=",")
      all_opts = sorted(list(set(dum_c4.columns).union(set(dum_all.columns))))

      for opt in all_opts:
        opt_label = clean_str(choices.get(opt, f"コード_{opt}"))
        cnt_c4 = dum_c4[opt].sum() if opt in dum_c4 else 0
        cnt_all = dum_all[opt].sum() if opt in dum_all else 0

        pct_c4 = (cnt_c4 / n_c4) * 100 if n_c4 > 0 else 0
        pct_all = (cnt_all / n_u30) * 100 if n_u30 > 0 else 0

        diff = pct_c4 - pct_all
        lift = (pct_c4 / pct_all) if pct_all > 0 else np.nan

        records.append({
            "大分類": cat,
            "設問コード": q_code,
            "設問名": title,
            "回答・選択肢": opt_label,
            "タイプ": "複数回答",
            "クラスタ4回答数": cnt_c4,
            "クラスタ4回答率(%)": round(pct_c4, 2),
            "U30全体回答率(%)": round(pct_all, 2),
            "比率差(%pt)": round(diff, 2),
            "偏り率(リフト値)": (
                round(lift, 2) if not np.isnan(lift) else "-"
            ),
        })

    # 数値項目 (N, NL)
    elif q_type in ["N", "NL"]:
      num_c4 = pd.to_numeric(s_c4, errors="coerce").dropna()
      num_all = pd.to_numeric(s_all, errors="coerce").dropna()

      if len(num_c4) > 0 and len(num_all) > 0:
        m_c4 = num_c4.mean()
        m_all = num_all.mean()
        diff = m_c4 - m_all
        lift = (m_c4 / m_all) if m_all != 0 else np.nan

        records.append({
            "大分類": cat,
            "設問コード": q_code,
            "設問名": title,
            "回答・選択肢": "【平均値】",
            "タイプ": "数値",
            "クラスタ4回答数": len(num_c4),
            "クラスタ4回答率(%)": round(m_c4, 2),
            "U30全体回答率(%)": round(m_all, 2),
            "比率差(%pt)": round(diff, 2),
            "偏り率(リフト値)": (
                round(lift, 2) if not np.isnan(lift) else "-"
            ),
        })

    # 単一選択 (S, SL, M)
    else:
      c4_counts = s_c4.astype(str).value_counts()
      all_counts = s_all.astype(str).value_counts()
      all_vals = [
          k for k in choices.keys() if k in c4_counts or k in all_counts
      ] or list(all_counts.index)

      for val in all_vals:
        if val in ["nan", "None", "", "0"]:
          continue
        opt_label = clean_str(choices.get(val, f"コード_{val}"))
        cnt_c4 = c4_counts.get(val, 0)
        cnt_all = all_counts.get(val, 0)

        pct_c4 = (cnt_c4 / n_c4) * 100 if n_c4 > 0 else 0
        pct_all = (cnt_all / n_u30) * 100 if n_u30 > 0 else 0

        diff = pct_c4 - pct_all
        lift = (pct_c4 / pct_all) if pct_all > 0 else np.nan

        records.append({
            "大分類": cat,
            "設問コード": q_code,
            "設問名": title,
            "回答・選択肢": opt_label,
            "タイプ": "単一回答",
            "クラスタ4回答数": cnt_c4,
            "クラスタ4回答率(%)": round(pct_c4, 2),
            "U30全体回答率(%)": round(pct_all, 2),
            "比率差(%pt)": round(diff, 2),
            "偏り率(リフト値)": (
                round(lift, 2) if not np.isnan(lift) else "-"
            ),
        })

  if (i + CHUNK_SIZE) % 500 < CHUNK_SIZE:
    print(
        f"  進捗: {min(i + CHUNK_SIZE, len(target_qs))} / {len(target_qs)} 問"
        " 完了...",
        flush=True,
    )

print("3. 結果を並び替えて保存中...", flush=True)
df_res = pd.DataFrame(records)
df_res.sort_values(by="比率差(%pt)", ascending=False, inplace=True)

df_res.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print("\n" + "=" * 70)
print(f"✅ 集計完了！全設問の偏り率ランキングを保存しました:\n{OUTPUT_CSV}")
print("=" * 70)

print("\n🔥 【クラスタ4で突出して高い（偏っている）項目 TOP 15】")
top_preview = df_res[df_res["クラスタ4回答率(%)"] >= 10.0].head(15)
cols_show = [
    "大分類",
    "設問名",
    "回答・選択肢",
    "クラスタ4回答率(%)",
    "U30全体回答率(%)",
    "比率差(%pt)",
    "偏り率(リフト値)",
]
print(top_preview[cols_show].to_string(index=False))