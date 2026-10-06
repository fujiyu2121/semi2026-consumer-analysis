# app.py
import json
import numpy as np
import pandas as pd
import requests
from scipy.stats import chi2_contingency
import streamlit as st

st.set_page_config(
    page_title="生活者データ分析ツール", layout="centered", page_icon="🥤"
)


# --- 1. 合言葉認証 ---
def check_password():
  if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

  if not st.session_state["authenticated"]:
    st.markdown("### 🔒 ゼミ分析ツール（班内限定）")
    user_name = st.text_input("あなたのお名前（記録用）", value="メンバー")
    pw = st.text_input("合言葉を入力してください", type="password")
    if st.button("ログイン"):
      if pw == "semi2026":
        st.session_state["authenticated"] = True
        st.session_state["user_name"] = user_name
        st.rerun()
      else:
        st.error("合言葉が違います")
    return False
  return True


if not check_password():
  st.stop()


# --- 2. データ読み込み ---
@st.cache_data
def load_data():
  df = pd.read_parquet("app_data.parquet")
  with open("question_master.json", "r", encoding="utf-8") as f:
    questions = json.load(f)
  return df, questions


df, questions = load_data()
total_n = len(df)

# 表示用の辞書（「日本語タイトル (設問コード)」の形式）
display_q_map = {q: f"{info['title']} ({q})" for q, info in questions.items()}
all_q_keys = list(questions.keys())


# 設問検索用関数
def search_questions(keyword):
  if not keyword:
    return all_q_keys[:50]
  kw = keyword.lower()
  hits = [k for k, v in questions.items() if kw in v["title"].lower()]
  return hits if hits else all_q_keys[:20]


st.title("🥤 ターゲット分析 & クロス集計")
st.caption(f"母集団全体 N = {total_n:,}人 ｜ ログイン中: {st.session_state.user_name}")

# --- 3. ターゲット設定（4変数以上） ---
st.subheader("🎯 1. ターゲット条件の設定")
st.write("各条件で設問と当てはまる回答（日本語）を選択してください。")

target_mask = pd.Series(True, index=df.index)
filter_summary = []

# 【条件①：性別】
col1, col2 = st.columns(2)
with col1:
  q1 = "AAF1" if "AAF1" in questions else all_q_keys[0]
  st.markdown(f"**条件①: {questions[q1]['title']}**")
with col2:
  opts1 = questions[q1]["choices"]
  # 選択肢の表示名（日本語）
  val1 = st.multiselect(
      "該当する性別",
      options=list(opts1.keys()),
      format_func=lambda x: opts1.get(x, x),
      default=["2"] if "2" in opts1 else [],  # デフォルト女性
      key="val1",
  )
  if val1:
    target_mask &= df[q1].astype(str).isin(val1)
    filter_summary.append(
        f"{questions[q1]['title']}: {[opts1.get(v, v) for v in val1]}"
    )

# 【条件②：年齢】
if "AAF2" in df.columns:
  st.markdown(f"**条件②: {questions.get('AAF2', {}).get('title', '年齢')}**")
  age_range = st.slider("対象年齢の範囲", 15, 79, (20, 29))
  target_mask &= (df["AAF2"] >= age_range[0]) & (df["AAF2"] <= age_range[1])
  filter_summary.append(f"年齢: {age_range[0]}〜{age_range[1]}歳")

# 【条件③：ライフスタイル・未既婚・職業等】
st.markdown("**条件③: 属性・ライフスタイル（設問を検索）**")
kw3 = st.text_input("条件③の質問を検索（例: 未既婚, 職業, 世帯）", value="未既婚")
matched3 = search_questions(kw3)
q3 = st.selectbox(
    "質問を選択",
    matched3,
    format_func=lambda x: display_q_map.get(x, x),
    key="q3",
)
opts3 = questions[q3]["choices"]
val3 = st.multiselect(
    f"「{questions[q3]['title']}」の該当項目",
    options=list(opts3.keys()),
    format_func=lambda x: opts3.get(x, x),
    key="val3",
)
if val3:
  target_mask &= df[q3].astype(str).isin(val3)
  filter_summary.append(
      f"{questions[q3]['title']}: {[opts3.get(v, v) for v in val3]}"
  )

# 【条件④：行動・価値観・利用頻度】
st.markdown("**条件④: 行動・価値観（設問を検索）**")
kw4 = st.text_input(
    "条件④の質問を検索（例: コンビニ, 美容, 健康, SNS）", value="コンビニ"
)
matched4 = search_questions(kw4)
q4 = st.selectbox(
    "質問を選択",
    matched4,
    format_func=lambda x: display_q_map.get(x, x),
    key="q4",
)
opts4 = questions[q4]["choices"]
val4 = st.multiselect(
    f"「{questions[q4]['title']}」の該当項目",
    options=list(opts4.keys()),
    format_func=lambda x: opts4.get(x, x),
    key="val4",
)
if val4:
  target_mask &= df[q4].astype(str).isin(val4)
  filter_summary.append(
      f"{questions[q4]['title']}: {[opts4.get(v, v) for v in val4]}"
  )

# ターゲット規模（市場サイズ）の表示
target_n = int(target_mask.sum())
target_ratio = (target_n / total_n) * 100 if total_n > 0 else 0

st.info(
    f"📊 **設定したターゲットの規模**\n\n"
    f"・対象人数: **{target_n:,} 人**\n\n"
    f"・全体比率: **{target_ratio:.2f}%**"
)

# --- 4. クロス集計・仮説検証 ---
st.divider()
st.subheader("🔍 2. 検証する質問とのクロス集計")
st.write("ターゲット群とその他の層で、回答比率に統計的差があるか検定します。")

kw_verify = st.text_input(
    "検証したい設問をキーワード検索（例: 美容, 健康, 新商品, 雑誌, Instagram）",
    value="美容",
)
matched_verify = search_questions(kw_verify)
target_q = st.selectbox(
    "検証する質問を選択",
    matched_verify,
    format_func=lambda x: display_q_map.get(x, x),
    key="verify_q",
)

q_info = questions[target_q]
q_title = q_info["title"]
choices = q_info["choices"]

# クロス集計データ作成
sub_df = pd.DataFrame({
    "グループ": np.where(target_mask, "ターゲット", "その他全体"),
    "回答": df[target_q].astype(str),
}).dropna()

# 存在しない無効コードや空白を除去
sub_df = sub_df[sub_df["回答"].isin(choices.keys())]

if len(sub_df) > 0 and len(sub_df["回答"].unique()) > 1 and target_n > 0:
  # 日本語ラベルに変換
  sub_df["回答ラベル"] = sub_df["回答"].map(choices)

  # 度数表 & 比率表（%）
  ct = pd.crosstab(sub_df["グループ"], sub_df["回答ラベル"])
  ct_pct = (
      pd.crosstab(sub_df["グループ"], sub_df["回答ラベル"], normalize="index")
      * 100
  )

  st.write(f"**【クロス集計結果: {q_title}（比率 %）】**")
  st.dataframe(ct_pct.round(1))

  # カイ二乗検定
  chi2, p, dof, expected = chi2_contingency(ct)

  col_m1, col_m2, col_m3 = st.columns(3)
  col_m1.metric("カイ二乗値 (χ²)", f"{chi2:.2f}")
  col_m2.metric("p値", f"{p:.4e}" if p < 0.0001 else f"{p:.4f}")
  col_m3.metric(
      "有意差の有無",
      "有意差あり (p<0.05)" if p < 0.05 else "差なし",
      delta="有意" if p < 0.05 else "なし",
  )
else:
  st.warning(
      "ターゲット該当者が0人か、回答データが存在しません。条件を見直してください。"
  )
  chi2, p = 0.0, 1.0

# --- 5. スプレッドシート保存 ---
st.divider()
st.subheader("📋 3. 班のスプレッドシートに保存")
GAS_URL = "ここにGASのウェブアプリURLを貼り付け"

target_desc = " ＆ ".join(filter_summary) if filter_summary else "全数"

if st.button("この分析結果をスプレッドシートに追記する"):
  if "ここにGAS" in GAS_URL:
    st.warning(
      "⚠️ まだGASのURLが設定されていません。スプレッドシート連携URLを設定してください。"
  )
  else:
    payload = {
        "user_name": st.session_state.get("user_name", "メンバー"),
        "target_name": target_desc,
        "sample_size": target_n,
        "total_ratio": f"{target_ratio:.2f}%",
        "question_title": q_title,
        "chi2": round(float(chi2), 2),
        "p_value": round(float(p), 4),
        "is_significant": "有意差あり(p<0.05)" if p < 0.05 else "有意差なし",
    }
    try:
      res = requests.post(GAS_URL, json=payload, timeout=5)
      if res.status_code == 200:
        st.success("🎉 スプレッドシートに記録されました！")
      else:
        st.error(f"送信エラー: {res.status_code}")
    except Exception as e:
      st.error(f"通信エラー: {e}")