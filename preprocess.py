# preprocess.py
import json
import pandas as pd

print("1. 設問マスターを読み込み、階層構造（大分類 -> 設問 -> 選択肢）を構築中...")
df_item = pd.read_csv("itemlist_SSC_19_4_encoding.csv", encoding="cp932")

categories = []
category_questions = {}  # {カテゴリ名: [設問コードリスト]}
questions = {}

current_cat = "その他"
current_q = None

for _, row in df_item.iterrows():
  q_type = str(row["Type"]).strip() if pd.notna(row["Type"]) else ""
  title = str(row["Title"]).strip() if pd.notna(row["Title"]) else ""
  q_code = str(row["Question"]).strip() if pd.notna(row["Question"]) else ""

  # Type == 'X' は大分類見出し
  if q_type.upper() == "X":
    current_cat = title
    if current_cat not in categories:
      categories.append(current_cat)
      category_questions[current_cat] = []
    current_q = None
    continue

  # 設問の定義行
  if pd.notna(row["Question"]) and q_code != "":
    current_q = q_code
    questions[current_q] = {
        "title": title,
        "category": current_cat,
        "type": q_type if q_type else "S",
        "choices": {},
    }
    if current_cat not in category_questions:
      categories.append(current_cat)
      category_questions[current_cat] = []
    if current_q not in category_questions[current_cat]:
      category_questions[current_cat].append(current_q)

  # 選択肢行
  elif current_q and pd.notna(row["CtgNo"]):
    try:
      ctg_no = str(int(row["CtgNo"]))
    except ValueError:
      ctg_no = str(row["CtgNo"]).strip()
    questions[current_q]["choices"][ctg_no] = title

print("2. データの整合性を確認中...")
# 選択肢が存在する設問、または KEY / 年齢などの重要列のみ抽出
sample_cols = pd.read_csv(
    "raw_SSC_19_4.csv", encoding="cp932", nrows=1
).columns.tolist()

valid_cols = [
    c
    for c in sample_cols
    if (c in questions and len(questions[c]["choices"]) > 0)
    or c in ["KEY", "AAF2"]
]

print(f"有効な列数: {len(valid_cols)} 列")

df_raw = pd.read_csv(
    "raw_SSC_19_4.csv",
    encoding="cp932",
    usecols=valid_cols,
    low_memory=False,
    dtype=str,
)

# preprocess.py の該当部分（マスター構築後）
# AAF2（年齢）など、選択肢を持たない数値列をマスターに登録
if "AAF2" in df_raw.columns:
  questions["AAF2"] = {
      "title": "年齢",
      "category": "基本属性",  # 基本属性カテゴリに配置
      "type": "NUM",  # 数値型フラグ
      "choices": {},
  }
  # カテゴリマップにも追加
  if "基本属性" in filtered_cat_questions:
    if "AAF2" not in filtered_cat_questions["基本属性"]:
      filtered_cat_questions["基本属性"].insert(0, "AAF2")

# 存在する列のみにマスターを絞り込み
filtered_questions = {k: questions[k] for k in valid_cols if k in questions}
filtered_cat_questions = {}
for cat, q_list in category_questions.items():
  available_qs = [q for q in q_list if q in filtered_questions]
  if available_qs:
    filtered_cat_questions[cat] = available_qs

master_payload = {
    "categories": list(filtered_cat_questions.keys()),
    "category_map": filtered_cat_questions,
    "questions": filtered_questions,
}

print("3. 軽量Parquet形式で保存中...")
df_raw.to_parquet("app_data.parquet", index=False, compression="snappy")

with open("question_master.json", "w", encoding="utf-8") as f:
  json.dump(master_payload, f, ensure_ascii=False, indent=2)

print("✅ 前処理が完了しました！")