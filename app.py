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


try:
  df, questions = load_data()
except Exception as e:
  st.error(f"データの読み込みに失敗しました: {e}")
  st.stop()

total_n = len(df)
all_q_keys = list(questions.keys())
display_q_map = {q: f"{info['title']} ({q})" for q, info in questions.items()}


# 設問検索用関数（0件ヒット時も安全に全件の一部を返す）
def search_questions(keyword):
  if not keyword or not keyword.strip():
    return all_q_keys[:50]
  kw = keyword.strip().lower()
  hits = [k for k, v in questions.items() if kw in v["title"].lower()]
  if not hits:
    return []
  return hits


st.title("🥤 ターゲット分析 & クロス集計")
st.caption(f"母集団全体 N = {total_n:,}人 ｜ ログイン中: {st.session_state.user_name}")

# --- 3. ターゲット設定（4変数以上） ---
st.subheader("🎯 1. ターゲット条件の設定")

target_mask = pd.Series(True, index=df.index)
filter_summary = []

# 【条件①：性別】
q1 = "AAF1" if "AAF1" in questions else all_q_keys[0]
opts1 = questions[q1].get("choices", {})
val1 = st.multiselect(
    f"条件①: {questions[q1]['title']}",
    options=list(opts1.keys()),
    format_func=lambda x: opts1.get(x, x),
    default=["2"] if "2" in opts1 else [],
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
  age_range = st.slider("対象年齢の範囲", 15, 79, (20, 29), key="age_slider")
  target_mask &= (df["AAF2"] >= age_range[0]) & (df["AAF2"] <= age_range[1])
  filter_summary.append(f"年齢: {age_range[0]}〜{age_range[1]}歳")

# 【条件③：属性・ライフスタイル】
st.markdown("**条件③: 属性・ライフスタイル（設問を検索）**")
kw3 = st.text_input(
    "質問キーワード（例: 未既婚, 職業, 世帯）", value="未既婚", key="kw3"
)
matched3 = search_questions(kw3)
if not matched3:
  st.warning(f"「{kw3}」に一致する設問が見つかりません。別の単語をお試しください。")
  matched3 = all_q_keys[:20]

q3 = st.selectbox(
    "質問を選択",
    matched3,
    format_func=lambda x: display_q_map.get(x, x),
    key="q3",
)
opts3 = questions[q3].get("choices", {})
val3 = st.multiselect(
    f"「{questions[q3]['title']}」の該当項目",
    options=list(opts3.keys()),
    format_func=lambda x: opts3.get(x, x),
    key=f"val3_{q3}",  # 設問が変わると選択肢も安全にリセット
)
if val3:
  target_mask &= df[q3].astype(str).isin(val3)
  filter_summary.append(
      f"{questions[q3]['title']}: {[opts3.get(v, v) for v in val3]}"
  )

# 【条件④：行動・価値観・利用頻度】
st.markdown("**条件④: 行動・価値観（設問を検索）**")
kw4 = st.text_input(
    "質問キーワード（例: コンビニ, 美容, 健康, SNS）", value="コンビニ", key="kw4"
)
matched4 = search_questions(kw4)
if not matched4:
  st.warning(f"「{kw4}」に一致する設問が見つかりません。別の単語をお試しください。")
  matched4 = all_q_keys[:20]

q4 = st.selectbox(
    "質問を選択",
    matched4,
    format_func=lambda x: display_q_map.get(x, x),
    key="q4",
)
opts4 = questions[q4].get("choices", {})
val4 = st.multiselect(
    f"「{questions[q4]['title']}」の該当項目",
    options=list(opts4.keys()),
    format_func=lambda x: opts4.get(x, x),
    key=f"val4_{q4}",
)
if val4:
  target_mask &= df[q4].astype(str).isin(val4)
  filter_summary.append(
      f"{questions[q4]['title']}: {[opts4.get(v, v) for v in val4]}"
  )

# ターゲット規模の表示
target_n = int(target_mask.sum())
target_ratio = (target_n / total_n) * 100 if total_n > 0 else 0

st.info(
    f"📊 **設定したターゲットの規模**\n\n"
    f"・対象人数: **{target_n:,} 人** ｜ 全体比率: **{target_ratio:.2f}%**"
)

# --- 4. クロス集計・仮説検証 ---
st.divider()
st.subheader("🔍 2. 検証する質問とのクロス集計")

kw_verify = st.text_input(
    "検証したい設問をキーワード検索（例: 美容, 買物, 飲料, 健康, 新商品）",
    value="美容",
    key="kw_verify",
)
matched_verify = search_questions(kw_verify)
if not matched_verify:
  st.warning(
      f"「{kw_verify}」に一致する設問が見つかりません。代表設問を表示します。"
  )
  matched_verify = all_q_keys[:30]

target_q = st.selectbox(
    "検証する質問を選択",
    matched_verify,
    format_func=lambda x: display_q_map.get(x, x),
    key="verify_q",
)

q_info = questions[target_q]
q_title = q_info["title"]
choices = q_info.get("choices", {})

chi2, p = 0.0, 1.0

if target_q in df.columns and len(choices) > 0 and target_n > 0:
  sub_df = pd.DataFrame({
      "グループ": np.where(target_mask, "ターゲット", "その他全体"),
      "回答": df[target_q].astype(str),
  }).dropna()

  # 選択肢マスターに定義された回答のみに限定
  sub_df = sub_df[sub_df["回答"].isin(choices.keys())]

  if len(sub_df) > 0 and len(sub_df["回答"].unique()) > 1:
    sub_df["回答ラベル"] = sub_df["回答"].map(choices)
    ct = pd.crosstab(sub_df["グループ"], sub_df["回答ラベル"])
    ct_pct = (
        pd.crosstab(sub_df["グループ"], sub_df["回答ラベル"], normalize="index")
        * 100
    )

    st.write(f"**【クロス集計結果: {q_title}（比率 %）】**")
    st.dataframe(ct_pct.round(1))

    # カイ二乗検定（例外防止）
    try:
      chi2, p, dof, expected = chi2_contingency(ct)
      col_m1, col_m2, col_m3 = st.columns(3)
      col_m1.metric("カイ二乗値 (χ²)", f"{chi2:.2f}")
      col_m2.metric("p値", f"{p:.4e}" if p < 0.0001 else f"{p:.4f}")
      col_m3.metric(
          "有意差の有無",
          "有意差あり (p<0.05)" if p < 0.05 else "有意差なし",
          delta="有意" if p < 0.05 else "なし",
      )
    except Exception as e:
      st.info(
          "※度数分布の偏りによりカイ二乗検定を実行できませんでした（十分なサンプル数または回答のばらつきが必要です）。"
      )
  else:
    st.warning("有効な回答データが不足しているため、クロス集計できません。")
else:
  st.warning(
      "ターゲット対象者が0人、または設問の選択肢情報が存在しません。条件を見直してください。"
  )

# --- 5. スプレッドシート保存 ---
st.divider()
st.subheader("📋 3. 班のスプレッドシートに保存")
GAS_URL = "ここにGASのウェブアプリURLを貼り付け"

target_desc = " ＆ ".join(filter_summary) if filter_summary else "全数"

if st.button("この分析結果をスプレッドシートに追記する"):
  if "ここにGAS" in GAS_URL:
    st.warning("⚠️ まだGASのWebアプリURLが設定されていません。")
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