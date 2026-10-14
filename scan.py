import pandas as pd
import numpy as np

# -------------------------------------------------------------
# 1. 設問名・選択肢名の辞書を作成（itemlistから軽量に作成）
# -------------------------------------------------------------
print("設問マスターを読み込み中...")
itemlist_path = r"itemlist_SSC_19_4_encoding.csv"

# エンコーディング対応
try:
    df_item = pd.read_csv(itemlist_path, encoding='cp932')
except Exception:
    df_item = pd.read_csv(itemlist_path, encoding='shift_jis')

q_titles = {}
q_choices = {}
current_q = None

for _, row in df_item.iterrows():
    q = str(row['Question']).strip() if pd.notna(row['Question']) else ""
    t = str(row['Title']).strip() if pd.notna(row['Title']) else ""
    c = row['CtgNo']
    
    if q != "":
        current_q = q
        q_titles[current_q] = t
    elif current_q and pd.notna(c):
        try:
            c_str = str(int(c))
        except Exception:
            c_str = str(c).strip()
        q_choices[(current_q, c_str)] = t

print(f"マスター登録完了: 設問数 {len(q_titles):,} 件, 選択肢数 {len(q_choices):,} 件")

# -------------------------------------------------------------
# 2. クラスター0のKEY（対象者ID）の読み込み
# -------------------------------------------------------------
print("クラスタリング結果を読み込み中...")
cluster_path = r"C:\Users\fujiy\semi2026\data-analysis\application\clustered_data.csv"

# もし clustered_data.csv が無ければ clustering_subset.csv から即時判定
try:
    df_cluster = pd.read_csv(cluster_path)
except Exception:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    subset_path = r"C:\Users\fujiy\semi2026\data-analysis\application\clustering_subset.csv"
    df_subset = pd.read_csv(subset_path)
    feature_cols = [
        'CBAA_33', 'CBAA_34', 'CCAB_2', 'CEAA_3', 'CEAA_4', 'CEAD_3',
        'CBAA_1', 'CBAA_4', 'CBAA_10', 'CBAA_29', 'CBAA_30',
        'CGAA_10', 'CGAA_11', 'CGAA_12',
        'CGAA_1', 'CGAA_2', 'CGAA_6', 'CGAA_7', 'CGAA_8'
    ]
    X = df_subset[[c for c in feature_cols if c in df_subset.columns]].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(X[col].median())
    X_scaled = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_subset['Cluster'] = kmeans.fit_predict(X_scaled)
    df_cluster = df_subset[['KEY', 'Cluster']]
    df_cluster.to_csv(cluster_path, index=False)

df_cluster['KEY'] = df_cluster['KEY'].astype(str)
c0_keys = set(df_cluster[df_cluster['Cluster'] == 0]['KEY'])
n_total = len(df_cluster)
n_c0 = len(c0_keys)
print(f"クラスター0: {n_c0:,}人 / 全体: {n_total:,}人 ({n_c0/n_total*100:.1f}%)")

# -------------------------------------------------------------
# 3. Parquetの全列を50列ずつバッチ集計（省メモリ設計）
# -------------------------------------------------------------
parquet_path = r"C:\Users\fujiy\semi2026\data-analysis\application\app_data.parquet"

# Parquet全体の列名を取得（KEYのみ読み出してcolumnsを取得）
all_columns = pd.read_parquet(parquet_path, columns=['KEY']).columns.tolist()
# 実際にはスキーマ情報から全列を取得
import pyarrow.parquet as pq
parquet_file = pq.ParquetFile(parquet_path)
all_columns = parquet_file.schema.names

target_cols = [c for c in all_columns if c not in ['KEY', 'SKEY'] and not c.startswith('DUMMY')]
print(f"スキャン対象の設問列数: {len(target_cols):,} 列")

batch_size = 50
results = []

print("全設問のスキャンを開始します...")
for i in range(0, len(target_cols), batch_size):
    cols_chunk = target_cols[i:i + batch_size]
    
    # チャンクごとにParquetからピンポイント読み込み
    df_chunk = pd.read_parquet(parquet_path, columns=['KEY'] + cols_chunk)
    df_chunk['KEY'] = df_chunk['KEY'].astype(str)
    mask_c0 = df_chunk['KEY'].isin(c0_keys)
    
    for q_code in cols_chunk:
        q_title = q_titles.get(q_code, q_code)
        
        # 年齢(AAF2)などの連続数値の場合
        if q_code == 'AAF2':
            s_num = pd.to_numeric(df_chunk[q_code], errors='coerce')
            tot_mean = s_num.mean()
            c0_mean = s_num[mask_c0].mean()
            results.append({
                '設問コード': q_code,
                '設問名': q_title,
                '回答コード': '-',
                '回答選択肢': '平均年齢',
                '全体割合(%)': round(tot_mean, 1),
                'クラスター0割合(%)': round(c0_mean, 1),
                '差分pt': round(c0_mean - tot_mean, 1),
                '倍率(リフト値)': round(c0_mean / tot_mean, 2) if tot_mean > 0 else 1.0,
                '全体人数': int(s_num.count()),
                'クラスター0人数': int(s_num[mask_c0].count())
            })
            continue

        # カテゴリ設問の場合（単一回答 or カンマ区切りの複数回答対応）
        series = df_chunk[q_code].dropna().astype(str)
        series_c0 = df_chunk.loc[mask_c0, q_code].dropna().astype(str)
        
        # カンマ区切り（複数回答）の展開集計
        tot_exploded = series.str.split(',').explode().str.strip()
        c0_exploded = series_c0.str.split(',').explode().str.strip()
        
        tot_cnts = tot_exploded.value_counts()
        c0_cnts = c0_exploded.value_counts()
        
        # 出現した回答コードを走査
        all_ans_codes = set(tot_cnts.index.tolist() + c0_cnts.index.tolist())
        
        for ans_code in all_ans_codes:
            if ans_code == '' or ans_code == 'nan':
                continue
                
            tot_c = tot_cnts.get(ans_code, 0)
            c0_c = c0_cnts.get(ans_code, 0)
            
            tot_rate = (tot_c / n_total) * 100
            c0_rate = (c0_c / n_c0) * 100
            
            # 全体出現率が極端に低い（1%未満）ゴミ項目は除外してノイズをカット
            if tot_rate < 1.0:
                continue
                
            diff = c0_rate - tot_rate
            lift = (c0_rate / tot_rate) if tot_rate > 0 else 0.0
            
            # 選択肢名を取得（辞書に無ければコードのまま）
            ans_clean = ans_code.split('.')[0]
            choice_label = q_choices.get((q_code, ans_clean), ans_code)
            
            results.append({
                '設問コード': q_code,
                '設問名': q_title,
                '回答コード': ans_code,
                '回答選択肢': choice_label,
                '全体割合(%)': round(tot_rate, 1),
                'クラスター0割合(%)': round(c0_rate, 1),
                '差分pt': round(diff, 1),
                '倍率(リフト値)': round(lift, 2),
                '全体人数': int(tot_c),
                'クラスター0人数': int(c0_c)
            })
            
    if (i + batch_size) % 500 == 0 or (i + batch_size) >= len(target_cols):
        print(f"進捗: {min(i + batch_size, len(target_cols)):,} / {len(target_cols):,} 列完了...")

# -------------------------------------------------------------
# 4. DataFrameに変換し、差分ポイント順にソートしてCSV出力
# -------------------------------------------------------------
df_result = pd.DataFrame(results)

# クラスター0で特に高い順（差分ptが大きい順）に並べ替え
df_result_sorted = df_result.sort_values(by='差分pt', ascending=False)

output_csv = r"C:\Users\fujiy\semi2026\data-analysis\application\cluster0_all_features_ranking.csv"
df_result_sorted.to_csv(output_csv, index=False, encoding='utf-8-sig')

print("\n" + "="*80)
print(f"🎉 完了！ 全 {len(df_result_sorted):,} 項目の集計結果を保存しました。")
print(f"ファイルパス: {output_csv}")
print("="*80)

# 上位15項目をコンソールに表示
print("\n【クラスター0の該当率が全体より高い質問項目 TOP 15】")
print(df_result_sorted[['設問名', '回答選択肢', '全体割合(%)', 'クラスター0割合(%)', '差分pt', '倍率(リフト値)']].head(15).to_string(index=False))