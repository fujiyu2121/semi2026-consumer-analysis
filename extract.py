import csv

input_path = r"C:\Users\fujiy\semi2026\data-analysis\application\raw_SSC_19_4.csv"
output_path = r"C:\Users\fujiy\semi2026\data-analysis\application\clustering_subset.csv"

# 抽出したい列名（クラスタリング用 + 比較用4問 + 個人識別KEY）
target_columns = {
    'KEY',
    # ① トレンド・SNS・映え意識
    'CBAA_33', 'CBAA_34', 'CCAB_2', 'CEAA_3', 'CEAA_4', 'CEAD_3',
    # ② プレミアム消費・プチ贅沢志向（300円受容性）
    'CBAA_1', 'CBAA_4', 'CBAA_10', 'CBAA_29', 'CBAA_30',
    # ③ 食のこだわり・楽しむ意識
    'CGAA_10', 'CGAA_11', 'CGAA_12',
    # ④ インナービューティ・健康意識
    'CGAA_1', 'CGAA_2', 'CGAA_6', 'CGAA_7', 'CGAA_8',
    # ⑤ 比較・検証用4問
    'CGAH', 'CFAK', 'CIAF_3', 'CIAF_01', 'CHAH'
}

# Shift-JIS / cp932 でストリーミング読み込み
encodings = ['cp932', 'shift_jis', 'utf-8-sig']

for enc in encodings:
    try:
        print(f"エンコーディング [{enc}] で読み込みを試みています...")
        with open(input_path, mode='r', encoding=enc, errors='replace') as infile:
            reader = csv.reader(infile)
            
            # 1行目（ヘッダー）の読み込み
            header = next(reader)
            print(f"全列数を検出: {len(header)} 列")
            
            # 必要な列のインデックス（何列目か）を記録
            col_indices = {}
            for idx, col_name in enumerate(header):
                clean_name = col_name.strip()
                if clean_name in target_columns:
                    col_indices[clean_name] = idx
            
            print(f"抽出対象として合致した列数: {len(col_indices)} 列")
            print("合致した列:", list(col_indices.keys()))
            
            # 抽出列の順番を固定
            selected_names = list(col_indices.keys())
            selected_idxs = [col_indices[name] for name in selected_names]
            
            # 軽量CSVへの書き出し
            with open(output_path, mode='w', encoding='utf-8-sig', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(selected_names)  # ヘッダー書き出し
                
                count = 0
                for row in reader:
                    # 必要なインデックスの値だけを抽出
                    extracted_row = [row[i] if i < len(row) else '' for i in selected_idxs]
                    writer.writerow(extracted_row)
                    count += 1
                    if count % 2000 == 0:
                        print(f"{count:,} 行を抽出中...")
            
            print(f"\n完了！ 全 {count:,} 行を '{output_path}' に抽出しました。")
            break
            
    except Exception as e:
        print(f"[{enc}] で失敗: {e}")