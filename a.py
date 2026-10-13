import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(
    os.path.join(BASE_DIR, "question_master.json"), "r", encoding="utf-8"
) as f:
  master = json.load(f)

# 例として、意識系の大分類から3問ピックアップして表示
sample_count = 0
for q_code, info in master["questions"].items():
  cat = info.get("category", "")
  if any(k in cat for k in ["生活", "意識", "価値観"]):
    print(f"【設問コード】: {q_code}")
    print(f"  大分類: {cat}")
    print(f"  タイプ: {info.get('type')}")
    print(f"  タイトル: {info.get('title')}")
    print(f"  選択肢(choices): {info.get('choices')}")
    print("-" * 50)
    sample_count += 1
    if sample_count >= 4:
      break