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
    master = json.load(f)
  return df, master


try:
  df, master = load_data()
  categories = master["categories"]
  cat_map = master["category_map"]
  questions = master["questions"]
except Exception as e:
  st.error(f"データの読み込みに失敗しました: {e}")
  st.stop()

total_n = len(df)

st.title("🥤 ターゲット分析 & クロス集計")
st.caption(f"母集団全体 N = {total_n:,}人 ｜ ログイン中: {st.session_state.user_name}")

# --- 3. ターゲット設定（自由な3段階ドリルダウン × 4条件） ---
st.subheader("🎯 1. ターゲット条件の設定（4変数）")
st.write(
    "大分類を選んでから設問を絞り込み、該当する回答を選択してください。"
)

target_mask = pd.Series(True, index=df.index)
filter_summary = []

# 条件入力コンポーネント（3段階ドリルダウン関数）
def render_condition_selector(idx):
  st.markdown(f"#### 【条件 {idx}】")
  c1, c2 = st.columns(2)

  with c1:
    selected_cat = st.selectbox(
        f"① 大分類を選択 (条件{idx})",
        options=categories,
        key=f"cat_{idx}",
    )

  available_qs = cat_map.get(selected_cat, [])
  if not available_qs:
    st.warning("この分類に含まれる有効な設問がありません。")
    return None, []

  with c2:
    selected_q = st.selectbox(
        f"② 質問を選択 (条件{idx})",
        options=available_qs,
        format_func=lambda q: f"{questions[q]['title']} ({q})",
        key=f"q_{idx}_{selected_cat}",  # 大分類変更で質問選択を安全に連動
    )

  q_info = questions[selected_q]
  choices = q_info.get("choices", {})

  if not choices:
    st.info("この質問には選択肢情報がありません。")
    return None, []

  # 選択肢のマルチセレクト
  selected_vals = st.multiselect(
      f"③ 該当する回答を選択（「{q_info['title']}」）",
      options=list(choices.keys()),
      format_func=lambda c: choices.get(c, c),
      key=f"val_{idx}_{selected_q}",
  )

  return selected_q, selected_vals


# 4つの条件をループ生成
for i in range(1, 5):
  q_code, vals = render_condition_selector(i)
  if q_code and vals and q_code in df.columns:
    target_mask &= df[q_code].astype(str).isin(vals)
    choice_labels = [questions[q_code]["choices"].get(v, v) for v in vals]
    filter_summary.append(
        f"{questions[q_code]['title']}: {', '.join(choice_labels)}"
    )

# ターゲット規模の計算と表示
target_n = int(target_mask.sum())
target_ratio = (target_n / total_n) * 100 if total_n > 0 else 0

st.divider()
st.info(
    f"📊 **設定したターゲットの規模**\n\n"
    f"・対象人数: **{target_n:,} 人** ｜ 全体比率: **{target_ratio:.2f}%**"
)

# --- 4. クロス集計・仮説検証（こちらも3段階ドリルダウンに統一） ---
st.divider()
st.subheader("🔍 2. 検証する質問とのクロス集計")
st.write(
    "比較・検証したい質問を大分類から選んでください（ターゲット層 vs"
    " その他全体）。"
)

v_col1, v_col2 = st.columns(2)
with v_col1:
  verify_cat = st.selectbox(
      "① 検証質問の大分類", options=categories, key="verify_cat"
  )

verify_available_qs = cat_map.get(verify_cat, [])
with v_col2:
  verify_q = st.selectbox(
      "② 検証する質問",
      options=verify_available_qs,
      format_func=lambda q: f"{questions[q]['title']} ({q})",
      key=f"verify_q_{verify_cat}",
  )

chi2, p = 0.0, 1.0
q_title = ""

if verify_q and verify_q in df.columns:
  q_info = questions[verify_q]
  q_title = q_info["title"]
  choices = q_info.get("choices", {})

  if len(choices) > 0 and target_n > 0:
    sub_df = pd.DataFrame({
        "グループ": np.where(target_mask, "ターゲット", "その他全体"),
        "回答": df[verify_q].astype(str),
    }).dropna()

    sub_df = sub_df[sub_df["回答"].isin(choices.keys())]

    if len(sub_df) > 0 and len(sub_df["回答"].unique()) > 1:
      sub_df["回答ラベル"] = sub_df["回答"].map(choices)
      ct = pd.crosstab(sub_df["グループ"], sub_df["回答ラベル"])
      ct_pct = (
          pd.crosstab(
              sub_df["グループ"], sub_df["回答ラベル"], normalize="index"
          )
          * 100
      )

      st.write(f"**【クロス集計結果: {q_title}（比率 %）】**")
      st.dataframe(ct_pct.round(1))

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
        st.info("※度数分布の偏りによりカイ二乗検定を実行できませんでした。")
    else:
      st.warning("有効な回答データが不足しているため、クロス集計できません。")
  else:
    st.warning("ターゲット対象者が0人、または選択肢情報が存在しません。")

# --- 5. スプレッドシート保存 ---
st.divider()
st.subheader("📋 3. 班のスプレッドシートに保存")
GAS_URL = "ここにGASのウェブアプリURLを貼り付け"

target_desc = " ＆ ".join(filter_summary) if filter_summary else "全数（条件なし）"

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