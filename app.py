# app.py
import json
import numpy as np
import pandas as pd
import requests
from scipy.stats import chi2_contingency
import streamlit as st

# --- ページ基本設定（スマホ対応） ---
st.set_page_config(
    page_title="生活者データ分析ツール", layout="centered", page_icon="📊"
)


# --- 1. 合言葉認証ゲート（班員以外をシャットアウト） ---
def check_password():
  if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

  if not st.session_state["authenticated"]:
    st.markdown("### 🔒 ゼミ分析ツール（班内限定）")
    user_name = st.text_input("あなたのお名前（スプレッドシート記録用）")
    pw = st.text_input("合言葉を入力してください", type="password")
    if st.button("ログイン"):
      if pw == "semi2026":  # 班内の合言葉
        st.session_state["authenticated"] = True
        st.session_state["user_name"] = user_name if user_name else "メンバー"
        st.rerun()
      else:
        st.error("合言葉が正しくありません")
    return False
  return True


if not check_password():
  st.stop()


# --- 2. データ読み込み（キャッシュ化して高速化） ---
@st.cache_data
def load_data():
  df = pd.read_parquet("app_data.parquet")
  with open("question_master.json", "r", encoding="utf-8") as f:
    questions = json.load(f)
  return df, questions


df, questions = load_data()
total_n = len(df)

st.title("🥤 ターゲット分析 & クロス集計")
st.caption(f"母集団全体 N = {total_n:,}人")

# --- 3. ターゲット作成（4条件以上のフィルタ） ---
st.subheader("1. ターゲット条件の設定")

# 選択しやすいように設問リストを整形
q_options = {q: f"{q}: {info['title']}" for q, info in questions.items()}
all_q_keys = list(q_options.keys())

# 条件1: 性別 (デフォルト AAF1)
target_mask = pd.Series(True, index=df.index)
filter_summary = []

col1, col2 = st.columns(2)
with col1:
  q1 = st.selectbox(
      "条件① 設問",
      all_q_keys,
      index=all_q_keys.index("AAF1") if "AAF1" in all_q_keys else 0,
      key="q1",
  )
with col2:
  opts1 = questions[q1]["choices"]
  val1 = st.multiselect(
      "条件① 該当選択肢",
      options=list(opts1.keys()),
      format_func=lambda x: opts1[x],
      key="val1",
  )
  if val1:
    target_mask &= df[q1].isin(val1)
    filter_summary.append(f"{questions[q1]['title']}: {[opts1[v] for v in val1]}")

# 条件2: 未既婚 (AAAB_01など)
with col1:
  q2 = st.selectbox(
      "条件② 設問",
      all_q_keys,
      index=all_q_keys.index("AAAB_01") if "AAAB_01" in all_q_keys else 1,
      key="q2",
  )
with col2:
  opts2 = questions[q2]["choices"]
  val2 = st.multiselect(
      "条件② 該当選択肢",
      options=list(opts2.keys()),
      format_func=lambda x: opts2[x],
      key="val2",
  )
  if val2:
    target_mask &= df[q2].isin(val2)
    filter_summary.append(f"{questions[q2]['title']}: {[opts2[v] for v in val2]}")

# 年齢絞り込み（AAF2）
if "AAF2" in df.columns:
  st.markdown("**条件③ 年齢層の指定**")
  age_range = st.slider("年齢範囲", 15, 79, (20, 29))
  target_mask &= (df["AAF2"] >= age_range[0]) & (df["AAF2"] <= age_range[1])
  filter_summary.append(f"年齢: {age_range[0]}〜{age_range[1]}歳")

# ターゲット規模の表示
target_n = int(target_mask.sum())
target_ratio = (target_n / total_n) * 100

st.info(
    f"🎯 **現在のターゲット規模**: **{target_n:,} 人** （全体の"
    f" **{target_ratio:.1f}%**）"
)

# --- 4. クロス集計 & カイ二乗検定 ---
st.subheader("2. 検証する質問を選択")
search_term = st.text_input(
    "質問を検索（例: 美容, コンビニ, SNS, 健康）", value="美容"
)

# 検索にヒットした設問
matched_keys = [
    k
    for k, v in questions.items()
    if search_term.lower() in v["title"].lower() and len(v["choices"]) > 0
]
if not matched_keys:
  matched_keys = all_q_keys[:10]

target_q = st.selectbox(
    "検証する質問項目", matched_keys, format_func=lambda x: q_options[x]
)
q_title = questions[target_q]["title"]
choices = questions[target_q]["choices"]

# クロス集計の実行
sub_df = pd.DataFrame({
    "Target": np.where(target_mask, "ターゲット", "その他"),
    "Answer": df[target_q],
}).dropna()

# 2×N クロス集計表
ct = pd.crosstab(sub_df["Target"], sub_df["Answer"])
ct_pct = (
    pd.crosstab(sub_df["Target"], sub_df["Answer"], normalize="index") * 100
)

# 選択肢のラベル付け
ct.columns = [choices.get(c, str(c)) for c in ct.columns]
ct_pct.columns = [choices.get(c, str(c)) for c in ct_pct.columns]

# カイ二乗検定
chi2, p, dof, expected = chi2_contingency(ct)

# 結果の表示
st.write(f"**【クロス集計結果 (比率 % )】**")
st.dataframe(ct_pct.round(1))

col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("カイ二乗値 (χ²)", f"{chi2:.2f}")
col_m2.metric("p値", f"{p:.4f}")
col_m3.metric(
    "統計的有意差",
    "あり (p<0.05) ★" if p < 0.05 else "なし",
    delta="有意" if p < 0.05 else "差なし",
)

# --- 5. スプレッドシートへ保存 ---
st.divider()
st.subheader("3. 班のスプレッドシートに保存")
GAS_URL = "ここにGASのウェブアプリURLを貼り付け"

target_desc = " / ".join(filter_summary)

if st.button("📋 この分析結果をシートに追記する"):
  if "ここにGAS" in GAS_URL:
    st.warning("先にGASのURLを設定してください。")
  else:
    payload = {
        "user_name": st.session_state.get("user_name", "メンバー"),
        "target_name": target_desc,
        "sample_size": target_n,
        "total_ratio": f"{target_ratio:.1f}%",
        "question_title": q_title,
        "target_pct": "詳細はシート参照",
        "chi2": round(float(chi2), 2),
        "p_value": round(float(p), 4),
        "is_significant": "有意差あり" if p < 0.05 else "差なし",
    }
    try:
      res = requests.post(GAS_URL, json=payload, timeout=5)
      if res.status_code == 200:
        st.success("スプレッドシートに保存が完了しました！")
      else:
        st.error(f"エラー: {res.status_code}")
    except Exception as e:
      st.error(f"通信に失敗しました: {e}")