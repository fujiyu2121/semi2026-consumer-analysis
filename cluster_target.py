# profile_cluster4.py
import json
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_PATH = os.path.join(BASE_DIR, "app_data.parquet")
CLUSTER_CSV = os.path.join(BASE_DIR, "u30_sns_target_clusters.csv")

# 1. クラスタデータの読み込み
df_clusters = pd.read_csv(CLUSTER_CSV)
df_clusters["KEY"] = df_clusters["KEY"].astype(str)

# 2. 属性列をParquetから読み込み
cols = ["KEY", "AAF1", "AAF2", "AAAA", "AAAB_01", "CGAN_1"]
df_all = pd.read_parquet(PARQUET_PATH, columns=cols)
df_all["KEY"] = df_all["KEY"].astype(str)

df = pd.merge(df_clusters, df_all, on="KEY", how="inner")
df["is_cluster4"] = df["Cluster"] == 3

c4 = df[df["is_cluster4"]]
all_u30 = df

print("=" * 60)
print(
    f"📊 【クラスタ4 プロファイル分析】 n={len(c4)} 人 (全体: {len(all_u30)} 人)"
)
print("=" * 60)

# 性別
print("\n▼ 性別比率:")
g_map = {"1": "男性", "2": "女性"}
s_c4_g = df.loc[df["is_cluster4"], "AAF1"].astype(str).map(g_map).value_counts(
    normalize=True
) * 100
s_all_g = df["AAF1"].astype(str).map(g_map).value_counts(normalize=True) * 100
comp_g = pd.DataFrame({"クラスタ4 (%)": s_c4_g, "30歳以下全体 (%)": s_all_g})
print(comp_g.round(1))

# 職業 TOP 5
print("\n▼ 職業構成 TOP 5:")
occ_map = {
    "1": "事務職",
    "2": "技術職",
    "3": "販売・サービス職",
    "4": "労務職",
    "7": "専門・自由業",
    "10": "高校生",
    "11": "大学生・院生",
    "14": "無職",
    "15": "その他",
}
s_c4_occ = df.loc[df["is_cluster4"], "AAAA"].astype(str).map(occ_map).value_counts(
    normalize=True
) * 100
s_all_occ = df["AAAA"].astype(str).map(occ_map).value_counts(normalize=True) * 100
comp_occ = pd.DataFrame(
    {"クラスタ4 (%)": s_c4_occ, "30歳以下全体 (%)": s_all_occ}
).head(5)
print(comp_occ.round(1))

# スタバ利用率
print("\n▼ スタバ利用頻度（月1日以上利用している割合）:")
sb_heavy_codes = ["1", "2", "3", "4", "5", "6"]
sb_c4 = (
    df.loc[df["is_cluster4"], "CGAN_1"].astype(str).isin(sb_heavy_codes).mean()
    * 100
)
sb_all = df["CGAN_1"].astype(str).isin(sb_heavy_codes).mean() * 100
print(f"  ・クラスタ4: {sb_c4:.1f}%")
print(f"  ・30歳以下全体: {sb_all:.1f}%")
print("=" * 60)