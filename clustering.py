import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ステップ1で作成した軽量CSVを指定
subset_path = r"C:\Users\fujiy\semi2026\data-analysis\application\clustering_subset.csv"
df = pd.read_csv(subset_path)
print(f"読み込み成功: {df.shape[0]} 行, {df.shape[1]} 列")

# 1. クラスタリング変数の選定
feature_cols = [
    'CBAA_33', 'CBAA_34', 'CCAB_2', 'CEAA_3', 'CEAA_4', 'CEAD_3',
    'CBAA_1', 'CBAA_4', 'CBAA_10', 'CBAA_29', 'CBAA_30',
    'CGAA_10', 'CGAA_11', 'CGAA_12',
    'CGAA_1', 'CGAA_2', 'CGAA_6', 'CGAA_7', 'CGAA_8'
]

existing_features = [c for c in feature_cols if c in df.columns]
X = df[existing_features].copy()

# 数値変換 & 欠損値補完（中央値）
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
    X[col] = X[col].fillna(X[col].median())

# 標準化 & k-means (k=4)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 2. 比較用4問のフラグ作成
# ① コンビニ（CGAH に 10 が含まれるか）
df['Target_1_CVS'] = df['CGAH'].astype(str).apply(lambda x: 1 if ('10' in [i.strip() for i in x.split(',')]) else 0) if 'CGAH' in df.columns else 0

# ② SNS真似買い（CFAK に 8 が含まれるか）
df['Target_2_SNS_Buy'] = df['CFAK'].astype(str).apply(lambda x: 1 if ('8' in [i.strip() for i in x.split(',')]) else 0) if 'CFAK' in df.columns else 0

# ③ 美容情報探索（CIAF_3 が 1 or 2、または CIAF_01 に 3 が含まれるか）
def check_beauty(row):
    if 'CIAF_3' in row and str(row['CIAF_3']).split('.')[0] in ['1', '2']:
        return 1
    if 'CIAF_01' in row and ('3' in [i.strip() for i in str(row['CIAF_01']).split(',')]):
        return 1
    return 0
df['Target_3_Beauty_Info'] = df.apply(check_beauty, axis=1)

# ④ サプリ・健康食品（CHAH に 9 が含まれるか）
df['Target_4_Supple'] = df['CHAH'].astype(str).apply(lambda x: 1 if ('9' in [i.strip() for i in x.split(',')]) else 0) if 'CHAH' in df.columns else 0

# 3. クラスター別 比較クロス集計の表示
cross_summary = df.groupby('Cluster')[[
    'Target_1_CVS', 'Target_2_SNS_Buy', 'Target_3_Beauty_Info', 'Target_4_Supple'
]].mean() * 100

cross_summary.columns = [
    '① コンビニ食品購入率(%)',
    '② SNS真似買い率(%)',
    '③ 美容新情報探索率(%)',
    '④ サプリ利用率(%)'
]

print("\n" + "="*50)
print("【クラスター別 ターゲット比較結果】")
print("="*50)
print(cross_summary.round(1))
print("="*50)

# クラスターごとの人数と構成比の集計
cluster_counts = df['Cluster'].value_counts().sort_index()
cluster_ratios = (df['Cluster'].value_counts(normalize=True).sort_index()) * 100

summary_df = pd.DataFrame({
    '人数 (人)': cluster_counts,
    '構成比 (%)': cluster_ratios.round(1)
})

print("\n" + "="*40)
print("【クラスター別 人数・構成比】")
print("="*40)
print(summary_df)
print(f"合計: {len(df):,} 人")
print("="*40)

import pandas as pd

# -------------------------------------------------------------
# 1. 設問コードと分かりやすい質問内容の対応辞書
# -------------------------------------------------------------
feature_labels = {
    # ① トレンド・SNS・自己表現
    'CBAA_33': '購入商品の写真や感想をSNS/ブログに投稿する',
    'CBAA_34': '満足した商品・サービスは積極的に人に伝えたい',
    'CCAB_2':  'ファッションを話題にしたりSNSに書き込む',
    'CEAA_3':  '異性から魅力的だと思われたい',
    'CEAA_4':  'SNSでフォロー・いいねを多く貰いたい',
    'CEAD_3':  '新商品・新しいコトの情報を自らチェックする',
    
    # ② プレミアム消費・買い物意識
    'CBAA_1':  'ラグジュアリー（最高級）ブランドが好きだ',
    'CBAA_4':  '価格が高くても品質の良いものを選ぶ',
    'CBAA_10': '新商品が出ると今持っているものを買い替えたくなる',
    'CBAA_29': 'お店でどの商品を買うか考えている時間が楽しい',
    'CBAA_30': 'リアル店舗で買い物することは楽しい',
    
    # ③ 食へのこだわり・食生活への投資
    'CGAA_10': 'おいしいものなど食生活を楽しむためにお金をかけたい',
    'CGAA_11': '料理の味やおいしさにこだわるほうだ',
    'CGAA_12': '高くても味や素材が良い料理を食べたい',
    
    # ④ インナービューティ・健康意識
    'CGAA_1':  '栄養バランスの良い食事をこころがけている',
    'CGAA_2':  '野菜や果物を十分に摂取している',
    'CGAA_6':  'カロリー控えめ・油分カットを積極的に選ぶ',
    'CGAA_7':  '低糖質・糖質制限をしている',
    'CGAA_8':  'デトックスやインナービューティ食材にひかれる'
}

# -------------------------------------------------------------
# 2. 各設問の「あてはまる（1または2）」フラグを集計
# -------------------------------------------------------------
# 回答値 1(非常によくあてはまる), 2(ややあてはまる) を 1、それ以外を 0 に変換
profile_summary = []

for col, label in feature_labels.items():
    if col in df.columns:
        # 1 または 2 に該当するか判定
        is_agree = df[col].isin([1, 2, 1.0, 2.0, '1', '2'])
        
        # 全体
        total_agree_cnt = is_agree.sum()
        total_agree_pct = (total_agree_cnt / len(df)) * 100
        
        row_dict = {
            '設問コード': col,
            '質問内容': label,
            '全体_人数': f"{total_agree_cnt:,}人",
            '全体_率': f"{total_agree_pct:.1f}%"
        }
        
        # クラスターごと（0〜3）
        for c in sorted(df['Cluster'].unique()):
            c_mask = (df['Cluster'] == c)
            c_total = c_mask.sum()
            c_agree_cnt = (is_agree & c_mask).sum()
            c_agree_pct = (c_agree_cnt / c_total) * 100
            
            row_dict[f'C{c}_人数'] = f"{c_agree_cnt:,}人"
            row_dict[f'C{c}_該当率'] = f"{c_agree_pct:.1f}%"
            
        profile_summary.append(row_dict)

df_profile = pd.DataFrame(profile_summary)

# -------------------------------------------------------------
# 3. 見やすく整形して表示 & CSV出力
# -------------------------------------------------------------
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 50)

print("\n" + "="*100)
print("【各クラスターの価値観・意識項目 該当者数と割合（あてはまる計：TOP2BOX）】")
print("="*100)
print(df_profile.to_string(index=False))
print("="*100)

# CSVに保存（Excelで開けるようにUTF-8 BOM付き）
df_profile.to_csv("cluster_detailed_profile.csv", index=False, encoding='utf-8-sig')
print("\n詳細プロファイル結果を 'cluster_detailed_profile.csv' に保存しました。")