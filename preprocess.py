# preprocess.py
import json
import pandas as pd

print("1. 設問マスターを読み込み中...")
df_item = pd.read_csv("itemlist_SSC_19_4_encoding.csv", encoding="cp932")

questions = {}
current_q = None

for _, row in df_item.iterrows():
  if pd.notna(row["Question"]):
    current_q = str(row["Question"]).strip()
    questions[current_q] = {
        "title": str(row["Title"]).strip(),
        "type": str(row["Type"]).strip() if pd.notna(row["Type"]) else "S",
        "choices": {},  # {"1": "男性", "2": "女性"} のように文字列キーで保持
    }
  elif current_q and pd.notna(row["CtgNo"]):
    ctg_str = str(int(row["CtgNo"]))
    questions[current_q]["choices"][ctg_str] = str(row["Title"]).strip()

print("2. 必要な列を抽出中...")
sample_cols = pd.read_csv(
    "raw_SSC_19_4.csv", encoding="cp932", nrows=1
).columns.tolist()

# 設問マスターに存在し、選択肢がある設問または重要設問
target_cols = [
    c
    for c in sample_cols
    if (c in questions and len(questions[c]["choices"]) > 0)
    or c in ["KEY", "AAF2"]
]

print(f"抽出対象の列数: {len(target_cols)} 列")

# データを文字列型として安全に読み込み
df_raw = pd.read_csv(
    "raw_SSC_19_4.csv",
    encoding="cp932",
    usecols=target_cols,
    low_memory=False,
    dtype=str,
)

# 年齢（AAF2）のみ数値変換
if "AAF2" in df_raw.columns:
  df_raw["AAF2"] = pd.to_numeric(df_raw["AAF2"], errors="coerce").fillna(0)

# 抽出した列のみのマスターを作成
filtered_questions = {k: questions[k] for k in target_cols if k in questions}

# preprocess.py の最後
print("3. Gzip圧縮CSVとマスターJSONを書き出し中...")
# compression='gzip' を指定するだけで自動で軽量圧縮されます
df_raw.to_csv("app_data.csv.gz", index=False, compression="gzip")

with open("question_master.json", "w", encoding="utf-8") as f:
  json.dump(filtered_questions, f, ensure_ascii=False, indent=2)

print("✅ 完了しました！ app_data.csv.gz を作成しました。")