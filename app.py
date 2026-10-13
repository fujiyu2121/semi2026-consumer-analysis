# app.py
import json
import os
import numpy as np
import pandas as pd
import requests
from scipy.stats import chi2_contingency, ttest_ind
import streamlit as st

st.set_page_config(
    page_title="生活者データ分析ツール", layout="wide", page_icon="🥤"
)

# --- 0. タイプ4潜在層キーの読み込み ---
ESTIMATE_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "type4_market_estimates.csv"
)


@st.cache_data
def load_type4_keys():
  if os.path.exists(ESTIMATE_CSV):
    df_est = pd.read_csv(ESTIMATE_CSV)
    df_est["KEY"] = df_est["KEY"].astype(str)
    # 30歳以下 かつ タイプ4確率 0.3以上のKEY
    mask = (df_est["is_u30"] == True) & (df_est["type4_prob"] >= 0.3)
    return set(df_est.loc[mask, "KEY"].unique())
  return set()


type4_keys = load_type4_keys()


# --- 1. 合言葉認証 ---
def check_password():
  if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

  if not st.session_state["authenticated"]:
    st.markdown("### 🔒 ゼミ分析ツール（班内限定）")
    user_name = st.text_input("あなたのお名前（記録用）", value="メンバー")
    pw = st.text_input("合言葉を入力してください", type="password")
    if st.button("ログイン"):
      if pw.strip() == "analysis":
        st.session_state["authenticated"] = True
        st.session_state["user_name"] = user_name
        st.rerun()
      else:
        st.error("合言葉が違います")
    return False
  return True


if not check_password():
  st.stop()


# --- 2. 設問マスター読み込み ---
@st.cache_data
def load_master():
  with open("question_master.json", "r", encoding="utf-8") as f:
    master = json.load(f)
  df_base = pd.read_parquet("app_data.parquet", columns=["KEY"])
  return master, len(df_base)


try:
  master, total_n = load_master()
  categories = master["categories"]
  cat_map = master["category_map"]
  questions = master["questions"]
except Exception as e:
  st.error(f"初期データの読み込みに失敗しました: {e}")
  st.stop()


def load_selected_columns(col_list):
  valid_cols = list(
      set([c for c in col_list if c and (c in questions or c == "KEY")])
  )
  if not valid_cols:
    valid_cols = ["KEY"]
  return pd.read_parquet("app_data.parquet", columns=valid_cols)


st.title("🥤 2グループ比較・ターゲットクロス集計")
st.caption(f"母集団全体 N = {total_n:,}人 ｜ ログイン中: {st.session_state.user_name}")

# --- 3. セッションステート初期化（条件IDの動的管理） ---
for grp in ["A", "B"]:
  if f"cond_ids_{grp}" not in st.session_state:
    st.session_state[f"cond_ids_{grp}"] = [1, 2]  # 初期は2個ずつ
  if f"next_id_{grp}" not in st.session_state:
    st.session_state[f"next_id_{grp}"] = 3


def add_condition(grp):
  st.session_state[f"cond_ids_{grp}"].append(st.session_state[f"next_id_{grp}"])
  st.session_state[f"next_id_{grp}"] += 1


def remove_condition(grp, cid):
  if len(st.session_state[f"cond_ids_{grp}"]) > 1:
    st.session_state[f"cond_ids_{grp}"].remove(cid)


# 条件セレクターUI関数
def render_condition(grp, cid, display_idx):
  h_col, del_col = st.columns([4, 1])
  with h_col:
    st.markdown(f"**条件 {display_idx}**")
  with del_col:
    if len(st.session_state[f"cond_ids_{grp}"]) > 1:
      if st.button("🗑️", key=f"del_{grp}_{cid}", help="この条件を削除"):
        remove_condition(grp, cid)
        st.rerun()

  sel_cat = st.selectbox(
      f"大分類 ({grp}-{display_idx})",
      options=categories,
      key=f"cat_{grp}_{cid}",
  )
  avail_qs = cat_map.get(sel_cat, [])
  if not avail_qs:
    return None, None

  sel_q = st.selectbox(
      f"質問 ({grp}-{display_idx})",
      options=avail_qs,
      format_func=lambda q: f"{questions[q]['title']} ({q})",
      key=f"q_{grp}_{cid}_{sel_cat}",
  )
  q_info = questions[sel_q]
  choices = q_info.get("choices", {})
  is_numeric = (
      q_info.get("type") == "NUM" or len(choices) == 0 or sel_q == "AAF2"
  )

  if is_numeric:
    val_range = st.slider(
        f"範囲（{q_info['title']}）",
        15,
        85,
        (20, 29),
        key=f"range_{grp}_{cid}_{sel_q}",
    )
    return sel_q, ("NUMERIC", val_range)
  else:
    vals = st.multiselect(
        f"回答（{q_info['title']}）",
        options=list(choices.keys()),
        format_func=lambda c: choices.get(c, c),
        key=f"val_{grp}_{cid}_{sel_q}",
    )
    return sel_q, ("CATEGORICAL", vals)


# --- 4. 2画面でターゲットA・Bを設定 ---
st.subheader("🎯 1. 比較する2つのターゲットを設定")
col_grp_a, col_grp_b = st.columns(2)

PRESET_OPTIONS = [
    "自由条件（設問から指定）",
    f"【プリセット】タイプ4潜在層（U30: {len(type4_keys):,}人）",
    "【プリセット】30歳以下全体（U30）",
]

active_conditions_A = []
with col_grp_a:
  st.markdown("### 🅰️ ターゲット A")
  target_a_mode = st.radio(
      "ターゲットAの指定方法", PRESET_OPTIONS, key="target_a_mode"
  )

  if target_a_mode == "自由条件（設問から指定）":
    for idx, cid in enumerate(st.session_state["cond_ids_A"], start=1):
      q_code, cond_val = render_condition("A", cid, idx)
      if q_code and cond_val:
        active_conditions_A.append((q_code, cond_val))
    st.button("➕ 条件を追加 (A)", on_click=add_condition, args=("A",))
  elif "タイプ4潜在層" in target_a_mode:
    st.success(f"🎯 タイプ4潜在層を選択中: {len(type4_keys):,} 人")
  else:
    st.info("👥 30歳以下全体を選択中")

active_conditions_B = []
with col_grp_b:
  st.markdown("### 🅱️