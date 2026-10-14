import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -------------------------------------------------------------
# 1. クラスター0のKEY（対象者ID）を特定
# -------------------------------------------------------------
print("クラスター0の対象者を抽出中...")
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

# クラスター0のKEYセットを作成
c0_keys = set(df_subset[df_subset['Cluster'] == 0]['KEY'].astype(str))
print(f"クラスター0の人数: {len(c0_keys):,} 人 / 全体: {len(df_subset):,} 人 ({len(c0_keys)/len(df_subset)*100:.1f}%)")

# -------------------------------------------------------------
# 2. JSONを使わずにParquetから直接集計する関数
# -------------------------------------------------------------
def inspect_question(q_code, q_label=""):
    """
    Parquetから指定列だけを読み込んで、全体 vs クラスター0 の集計を行う
    """
    parquet_path = r"C:\Users\fujiy\semi2026\data-analysis\application\app_data.parquet"
    
    try:
        # 指定列とKEYの2列だけを読み込む（メモリ消費は数MB以下）
        df_target = pd.read_parquet(parquet_path, columns=["KEY", q_code])
    except Exception as e:
        print(f"\n⚠️ [{q_code}] の読み込みに失敗しました: {e}")
        return

    df_target['KEY'] = df_target['KEY'].astype(str)
    df_target['Is_Cluster0'] = df_target['KEY'].isin(c0_keys)

    print("\n" + "="*60)
    print(f"【集計結果】 設問: {q_label} ({q_code})")
    print("="*60)

    # 年齢（AAF2）などの連続数値の場合
    if q_code == 'AAF2':
        s_num = pd.to_numeric(df_target[q_code], errors='coerce')
        c0_mean = s_num[df_target['Is_Cluster0']].mean()
        tot_mean = s_num.mean()
        print(f"・全体 平均値       : {tot_mean:.1f} 歳")
        print(f"・クラスター0 平均値 : {c0_mean:.1f} 歳")
        return

    # 選択肢（回答コード）ごとの集計
    tot_counts = df_target[q_code].value_counts(dropna=True)
    tot_pct = (tot_counts / len(df_target)) * 100

    c0_counts = df_target[df_target['Is_Cluster0']][q_code].value_counts(dropna=True)
    c0_pct = (c0_counts / len(c0_keys)) * 100

    # まとめて比較表を作成
    codes = sorted(list(set(tot_counts.index.tolist() + c0_counts.index.tolist())), key=lambda x: str(x))
    
    res = []
    for c in codes:
        t_p = tot_pct.get(c, 0.0)
        c_p = c0_pct.get(c, 0.0)
        diff = c_p - t_p
        lift = (c_p / t_p) if t_p > 0 else 0.0
        res.append({
            '回答コード': c,
            '全体(%)': round(t_p, 1),
            'クラスター0(%)': round(c_p, 1),
            '差分pt': round(diff, 1),
            '倍率(リフト)': round(lift, 2)
        })

    df_res = pd.DataFrame(res).sort_values(by='差分pt', ascending=False)
    print(df_res.to_string(index=False))

# -------------------------------------------------------------
# 3. 気になる設問を指定して実行
# -------------------------------------------------------------
# 調べたい設問コードと、わかりやすい日本語名を並べるだけ
targets = [
    ('AAF1', '性別 (1:男性, 2:女性)'),
    ('AAF2', '年齢'),
    ('CGAJ', 'ふだん飲む飲料ジャンル'),
    ('HDAB_4', 'Instagram利用頻度'),
    ('HDAB_2', 'Twitter利用頻度'),
]

for code, label in targets:
    inspect_question(code, label)