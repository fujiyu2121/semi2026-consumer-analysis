import pandas as pd
import numpy as np
import math
import pyarrow.parquet as pq

# -------------------------------------------------------------
# 0. 超高速・省メモリな 2x2 カイ二乗検定関数 (自由度1)
# -------------------------------------------------------------
def calc_chi2_2x2(a, b, c, d):
    """
    分割表:
                  該当       非該当      合計
    クラスター0   a          b          a+b (n_c0)
    対照群(他)   c          d          c+d (n_other)
    合計          a+c        b+d        N
    """
    N = a + b + c + d
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    if denom == 0:
        return 0.0, 1.0
    
    # ピアソンのカイ二乗値
    chi2 = (N * ((a * d - b * c) ** 2)) / denom
    
    # 自由度1における正確なp値: erfc(sqrt(chi2 / 2))
    p_val = math.erfc(math.sqrt(chi2 / 2.0))
    return float(chi2), float(p_val)

# -------------------------------------------------------------
# 1. 設問マスター辞書の読み込み
# -------------------------------------------------------------
print("設問マスターを読み込み中...")
itemlist_path = r"itemlist_SSC_19_4_encoding.csv"

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

df_cluster = pd.read_csv(cluster_path)
df_cluster['KEY'] = df_cluster['KEY'].astype(str)

c0_keys = set(df_cluster[df_cluster['Cluster'] == 0]['KEY'])
n_total = len(df_cluster)
n_c0 = len(c0_keys)
n_other = n_total - n_c0
print(f"クラスター0: {n_c0:,}人 / 対照群(その他): {n_other:,}人 / 全体: {n_total:,}人 ({n_c0/n_total*100:.1f}%)")

# -------------------------------------------------------------
# 3. Parquetの全列を50列ずつバッチ集計 & カイ二乗検定
# -------------------------------------------------------------
parquet_path = r"C:\Users\fujiy\semi2026\data-analysis\application\app_data.parquet"

parquet_file = pq.ParquetFile(parquet_path)
all_columns = parquet_file.schema.names

target_cols = [c for c in all_columns if c not in ['KEY', 'SKEY'] and not c.startswith('DUMMY')]
print(f"スキャン対象の設問列数: {len(target_cols):,} 列")

batch_size = 50
results = []

print("全設問のスキャンと有意差検定を開始します...")
for i in range(0, len(target_cols), batch_size):
    cols_chunk = target_cols[i:i + batch_size]
    
    # チャンクごとにParquetからピンポイント読み込み
    df_chunk = pd.read_parquet(parquet_path, columns=['KEY'] + cols_chunk)
    df_chunk['KEY'] = df_chunk['KEY'].astype(str)
    mask_c0 = df_chunk['KEY'].isin(c0_keys)
    
    for q_code in cols_chunk:
        q_title = q_titles.get(q_code, q_code)
        
        # 年齢(AAF2)などの連続数値の場合（t検定ベース）
        if q_code == 'AAF2':
            s_num = pd.to_numeric(df_chunk[q_code], errors='coerce')
            tot_mean = s_num.mean()
            c0_mean = s_num[mask_c0].mean()
            other_mean = s_num[~mask_c0].mean()
            
            results.append({
                '設問コード': q_code,
                '設問名': q_title,
                '回答コード': '-',
                '回答選択肢': '平均年齢',
                '全体割合(%)': round(tot_mean, 1),
                'クラスター0割合(%)': round(c0_mean, 1),
                '対照群割合(%)': round(other_mean, 1),
                '差分pt': round(c0_mean - tot_mean, 1),
                '倍率(リフト値)': round(c0_mean / tot_mean, 2) if tot_mean > 0 else 1.0,
                '全体人数': int(s_num.count()),
                'クラスター0人数': int(s_num[mask_c0].count()),
                'カイ二乗値': np.nan,
                'p値': 0.0,
                '有意判定(5%水準)': '有意差あり (p<0.05)'
            })
            continue

        # カテゴリ設問の場合（単一回答 or カンマ区切りの複数回答対応）
        series_tot = df_chunk[q_code].dropna().astype(str)
        series_c0 = df_chunk.loc[mask_c0, q_code].dropna().astype(str)
        series_other = df_chunk.loc[~mask_c0, q_code].dropna().astype(str)
        
        tot_exploded = series_tot.str.split(',').explode().str.strip()
        c0_exploded = series_c0.str.split(',').explode().str.strip()
        other_exploded = series_other.str.split(',').explode().str.strip()
        
        tot_cnts = tot_exploded.value_counts()
        c0_cnts = c0_exploded.value_counts()
        other_cnts = other_exploded.value_counts()
        
        all_ans_codes = set(tot_cnts.index.tolist() + c0_cnts.index.tolist())
        
        for ans_code in all_ans_codes:
            if ans_code in ['', 'nan']:
                continue
                
            tot_c = tot_cnts.get(ans_code, 0)
            c0_c = c0_cnts.get(ans_code, 0)
            other_c = other_cnts.get(ans_code, 0)
            
            tot_rate = (tot_c / n_total) * 100
            c0_rate = (c0_c / n_c0) * 100
            other_rate = (other_c / n_other) * 100
            
            # 全体出現率が1%未満のノイズ項目はスキップ
            if tot_rate < 1.0:
                continue
                
            diff = c0_rate - tot_rate
            lift = (c0_rate / tot_rate) if tot_rate > 0 else 0.0
            
            # 2x2分割表のセル値計算
            # クラスター0: 該当=a, 非該当=b
            # 対照群(その他): 該当=c, 非該当=d
            a = c0_c
            b = n_c0 - c0_c
            c = other_c
            d = n_other - other_c
            
            chi2_val, p_val = calc_chi2_2x2(a, b, c, d)
            is_sig = "有意差あり (p<0.05)" if p_val < 0.05 else "有意差なし"
            
            ans_clean = ans_code.split('.')[0]
            choice_label = q_choices.get((q_code, ans_clean), ans_code)
            
            results.append({
                '設問コード': q_code,
                '設問名': q_title,
                '回答コード': ans_code,
                '回答選択肢': choice_label,
                '全体割合(%)': round(tot_rate, 1),
                'クラスター0割合(%)': round(c0_rate, 1),
                '対照群割合(%)': round(other_rate, 1),
                '差分pt': round(diff, 1),
                '倍率(リフト値)': round(lift, 2),
                '全体人数': int(tot_c),
                'クラスター0人数': int(c0_c),
                'カイ二乗値': round(chi2_val, 2),
                'p値': f"{p_val:.4e}" if p_val < 0.0001 else round(p_val, 4),
                '有意判定(5%水準)': is_sig
            })
            
    if (i + batch_size) % 500 == 0 or (i + batch_size) >= len(target_cols):
        print(f"進捗: {min(i + batch_size, len(target_cols)):,} / {len(target_cols):,} 列完了...")

# -------------------------------------------------------------
# 4. DataFrameに変換してCSV保存
# -------------------------------------------------------------
df_result = pd.DataFrame(results)

# 差分ptが大きい順（クラスター0で突出して高い項目順）にソート
df_result_sorted = df_result.sort_values(by='差分pt', ascending=False)

output_csv = r"C:\Users\fujiy\semi2026\data-analysis\application\cluster0_all_features_ranking.csv"
df_result_sorted.to_csv(output_csv, index=False, encoding='utf-8-sig')

print("\n" + "="*80)
print(f"🎉 完了！ 全 {len(df_result_sorted):,} 項目の検定付き集計結果を保存しました。")
print(f"保存先: {output_csv}")
print("="*80)

# プレビュー表示
preview_cols = ['設問名', '回答選択肢', '全体割合(%)', 'クラスター0割合(%)', '差分pt', 'カイ二乗値', 'p値', '有意判定(5%水準)']
print("\n【クラスター0突出項目 TOP 15（カイ二乗検定付き）】")
print(df_result_sorted[preview_cols].head(15).to_string(index=False))