import csv
import pandas as pd
import numpy as np

# -------------------------------------------------------------
# 1. クラスター0に属する人の KEY（ID）リストを取得
# -------------------------------------------------------------
subset_path = r"C:\Users\fujiy\semi2026\data-analysis\application\clustering_subset.csv"
# 直前のクラスタリング結果が含まれるDataFrameからKEYを抽出
# （※もしCluster列を保存していなければ、ここで再付与）
df_subset = pd.read_csv(subset_path)

# クラスタリングを再現してクラスター0のKEYを特定
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

feature_cols = [
    'CBAA_33', 'CBAA_34', 'CCAB_2', 'CEAA_3', 'CEAA_4', 'CEAD_3',
    'CBAA_1', 'CBAA_4', 'CBAA_10', 'CBAA_29', 'CBAA_30',
    'CGAA_10', 'CGAA_11', 'CGAA_12',
    'CGAA_1', 'CGAA_2', 'CGAA_6', 'CGAA_7', 'CGAA_8'
]
X = df_subset[[c for c in feature_cols if c in df_subset.columns]].copy()
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(X[col].median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_subset['Cluster'] = kmeans.fit_predict(X_scaled)

# クラスター0のKEYのセット
c0_keys = set(df_subset[df_subset['Cluster'] == 0]['KEY'].astype(str))
n_c0 = len(c0_keys)
n_total = len(df_subset)
print(f"クラスター0の対象者数: {n_c0:,}人 / 全体: {n_total:,}人 ({n_c0/n_total*100:.1f}%)")

# -------------------------------------------------------------
# 2. 質問項目の辞書（itemlist）を読み込み
# -------------------------------------------------------------
itemlist_path = r"itemlist_SSC_19_4_encoding.csv" # 同一フォルダまたはフルパス
try:
    df_itemlist = pd.read_csv(itemlist_path, encoding='cp932')
except Exception:
    df_itemlist = pd.read_csv(itemlist_path, encoding='shift_jis')

# Questionコードと質問タイトルの対応マップ
item_dict = dict(zip(df_itemlist['Question'].dropna().astype(str), df_itemlist['Title'].dropna().astype(str)))

# -------------------------------------------------------------
# 3. 生データを1行ずつ読み、全設問の「あてはまり数」をカウント
# -------------------------------------------------------------
raw_path = r"C:\Users\fujiy\semi2026\data-analysis\application\raw_SSC_19_4.csv"

# 各列の肯定回答（1または2）カウント用
total_counts = {}
c0_counts = {}

print("全設問のスキャンを開始します（Pandasを使わずストリーミング集計）...")

with open(raw_path, mode='r', encoding='cp932', errors='replace') as infile:
    reader = csv.reader(infile)
    header = [col.strip() for col in next(reader)]
    
    key_idx = header.index('KEY') if 'KEY' in header else 0
    
    for col in header:
        total_counts[col] = 0
        c0_counts[col] = 0
        
    line_num = 0
    for row in reader:
        line_num += 1
        is_c0 = str(row[key_idx]).strip() in c0_keys
        
        for idx, val in enumerate(row):
            col_name = header[idx]
            val_clean = val.strip()
            
            # 1または2（あてはまる）、あるいは複数回答(MA)で選択されている場合
            # 数値1, 2の判定、または空文字でない判定
            if val_clean in ['1', '2', '1.0', '2.0']:
                total_counts[col_name] += 1
                if is_c0:
                    c0_counts[col_name] += 1
                    
        if line_num % 3000 == 0:
            print(f"{line_num:,} 人分をスキャン完了...")

print("集計完了！ 特徴項目のスコアリングを計算中...")

# -------------------------------------------------------------
# 4. 全体比でクラスター0が特に高い「特徴項目」を抽出
# -------------------------------------------------------------
records = []
for col in header:
    if col in ['KEY', 'SKEY'] or col.startswith('DUMMY'):
        continue
        
    tot_cnt = total_counts[col]
    c0_cnt = c0_counts[col]
    
    tot_rate = (tot_cnt / n_total) * 100
    c0_rate = (c0_cnt / n_c0) * 100
    
    # サンプル数が少なすぎる（全体で5%未満）設問は除外
    if tot_rate < 5.0:
        continue
        
    # リフト値（倍率 = クラスター0率 / 全体率）と差分（ポイント差）
    lift = c0_rate / tot_rate if tot_rate > 0 else 0
    diff = c0_rate - tot_rate
    
    title = item_dict.get(col, '')
    
    records.append({
        '設問コード': col,
        '質問内容': title,
        '全体率(%)': round(tot_rate, 1),
        'クラスター0率(%)': round(c0_rate, 1),
        '差分pt': round(diff, 1),
        '倍率(リフト値)': round(lift, 2)
    })

df_res = pd.DataFrame(records)

# 差分が大きい順（クラスター0で圧倒的に高い項目）
top_diff = df_res.sort_values(by='差分pt', ascending=False).head(30)
# 倍率が大きい順
top_lift = df_res.sort_values(by='倍率(リフト値)', ascending=False).head(30)

print("\n" + "="*80)
print("【クラスター0が全体より圧倒的に高い設問 TOP 20（差分pt順）】")
print("="*80)
print(top_diff[['設問コード', '質問内容', '全体率(%)', 'クラスター0率(%)', '差分pt']].head(20).to_string(index=False))

# CSVに全スコアを保存
df_res.sort_values(by='差分pt', ascending=False).to_csv(
    r"C:\Users\fujiy\semi2026\data-analysis\application\cluster0_all_item_ranking.csv",
    index=False,
    encoding='utf-8-sig'
)
print("\n全質問の比較結果を 'cluster0_all_item_ranking.csv' に出力しました。")