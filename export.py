# export_type4_data.py
import json
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_PATH = os.path.join(BASE_DIR, "app_data.parquet")
ESTIMATE_CSV = os.path.join(BASE_DIR, "type4_market_estimates.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "type4_potential_u30_users.csv")

# 1. 先ほどの推計結果から「30歳以下 かつ 閾値0.3以上」の対象者を特定
if not os.path.exists(ESTIMATE_CSV):
  print(f"エラー: {ESTIMATE_CSV} が見つかりません。先に estimate.py を実行してください。")
  exit()

df_est = pd.read_csv(ESTIMATE_CSV)
df_est["KEY"] = df_est["KEY"].astype(str)

# 条件: 30歳以下 かつ type4_prob >= 0.3
target_mask = (df_est["is_u30"] == True) & (df_est["type4_prob"] >= 0.3)
df_target_keys = df_est.loc[target_mask, ["KEY", "AAF2", "CGAN_1", "is_starbucks_user", "type4_prob"]].copy()

n_target = len(df_target_keys)
print(f"🎯 抽出対象（30歳以下 × タイプ4確率0.3以上）: {n_target} 人")

# 2. Parquetから対象者の全設問データを抽出
print("Parquetから全回答データを読み込み中...", flush=True)
df_all = pd.read_parquet(PARQUET_PATH)
df_all["KEY"] = df_all["KEY"].astype(str)

# 対象者のみに絞り込み
df_target_full = pd.merge(df_target_keys, df_all, on="KEY", how="inner")

# 3. CSVとして保存（Excel対応 utf-8-sig）
try:
  df_target_full.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
  print("\n" + "=" * 65)
  print(f"✅ 抽出完了！ファイルを保存しました:")
  print(f"   {OUTPUT_CSV} ({df_target_full.shape[0]}行 × {df_target_full.shape[1]}列)")
  print("=" * 65)
except PermissionError:
  alt_output = os.path.join(BASE_DIR, "type4_potential_u30_users_new.csv")
  df_target_full.to_csv(alt_output, index=False, encoding="utf-8-sig")
  print(f"⚠️ ファイルが開かれていたため別名で保存しました: {alt_output}")