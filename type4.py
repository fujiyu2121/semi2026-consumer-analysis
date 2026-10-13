# create_type4_json.py
import json
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "u30_sns_target_clusters.csv")
JSON_PATH = os.path.join(BASE_DIR, "type4_keys.json")

# クラスタ結果CSVの読み込み
df = pd.read_csv(CSV_PATH)

# クラスタ 4 (値は 3) のKEYを抽出
target_keys = df.loc[df["Cluster"] == 3, "KEY"].astype(str).tolist()

# type4_keys.json として書き出し
with open(JSON_PATH, "w", encoding="utf-8") as f:
  json.dump(target_keys, f)

print(f"✅ 作成完了: {JSON_PATH} に {len(target_keys)} 人分を書き出しました！")