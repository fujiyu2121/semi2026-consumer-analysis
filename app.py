# app.py
import json
import numpy as np
import pandas as pd
import requests
from scipy.stats import chi2_contingency, ttest_ind
import streamlit as st

st.set_page_config(
    page_title="生活者データ分析ツール", layout="wide", page_icon="🥤"
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

active_conditions_A = []
with col_grp_a:
  st.markdown("### 🅰️ ターゲット A")
  for idx, cid in enumerate(st.session_state["cond_ids_A"], start=1):
    q_code, cond_val = render_condition("A", cid, idx)
    if q_code and cond_val:
      active_conditions_A.append((q_code, cond_val))
  st.button("➕ 条件を追加 (A)", on_click=add_condition, args=("A",))

active_conditions_B = []
with col_grp_b:
  st.markdown("### 🅱️ ターゲット B（対照群）")
  for idx, cid in enumerate(st.session_state["cond_ids_B"], start=1):
    q_code, cond_val = render_condition("B", cid, idx)
    if q_code and cond_val:
      active_conditions_B.append((q_code, cond_val))
  st.button("➕ 条件を追加 (B)", on_click=add_condition, args=("B",))

# --- 5. 検証質問の選択 ---
st.divider()
st.subheader("🔍 2. 比較・検証する質問の選択")

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

focus_choices = []
if verify_q:
  q_info = questions[verify_q]
  choices = q_info.get("choices", {})
  is_verify_numeric = (
      q_info.get("type") == "NUM" or len(choices) == 0 or verify_q == "AAF2"
  )

  if not is_verify_numeric and len(choices) > 0:
    focus_choices = st.multiselect(
        "③ 注目する回答を選択（任意：選択すると該当回答 vs それ以外に2値化されます）",
        options=list(choices.keys()),
        format_func=lambda c: choices.get(c, c),
        key=f"focus_{verify_q}",
    )

# --- 6. オンデマンド読み込みと集計 ---
needed_cols = [c[0] for c in active_conditions_A] + [
    c[0] for c in active_conditions_B
]
if verify_q:
  needed_cols.append(verify_q)

df_active = load_selected_columns(needed_cols)


# マスク作成関数
def make_mask(conds, df):
  mask = pd.Series(True, index=df.index)
  summary = []
  for q_code, cond_data in conds:
    mode, val = cond_data
    if q_code in df.columns:
      if mode == "NUMERIC":
        s_num = pd.to_numeric(df[q_code], errors="coerce")
        mask &= (s_num >= val[0]) & (s_num <= val[1])
        summary.append(f"{questions[q_code]['title']}: {val[0]}〜{val[1]}")
      else:
        if val:
          mask &= df[q_code].astype(str).isin(val)
          labels = [questions[q_code]["choices"].get(v, v) for v in val]
          summary.append(
              f"{questions[q_code]['title']}: {', '.join(labels)}"
          )
  return mask, summary


mask_A, summary_A = make_mask(active_conditions_A, df_active)
mask_B, summary_B = make_mask(active_conditions_B, df_active)

nA = int(mask_A.sum())
nB = int(mask_B.sum())

# 規模サマリー
info_col1, info_col2 = st.columns(2)
with info_col1:
  st.info(
      f"🅰️ **ターゲット A 規模**\n\n・人数: **{nA:,} 人** ("
      f" {(nA/total_n)*100:.2f}%)"
  )
with info_col2:
  st.info(
      f"🅱️ **ターゲット B 規模**\n\n・人数: **{nB:,} 人** ("
      f" {(nB/total_n)*100:.2f}%)"
  )

test_stat, p_val = 0.0, 1.0
q_title = ""
is_sig_str = "有意差なし"

# 集計実行
if verify_q and verify_q in df_active.columns and nA > 0 and nB > 0:
  q_info = questions[verify_q]
  q_title = q_info["title"]
  choices = q_info.get("choices", {})
  is_verify_numeric = (
      q_info.get("type") == "NUM" or len(choices) == 0 or verify_q == "AAF2"
  )

  # AとBの比較対象データフレーム作成
  # （両方に該当するサンプルは重複を避けるため除外またはA優先処理）
  idx_A = df_active[mask_A & ~mask_B].index
  idx_B = df_active[mask_B & ~mask_A].index

  if len(idx_A) == 0 or len(idx_B) == 0:
    st.warning(
        "ターゲットAとターゲットBが完全に重複しているか、対象者が0人です。条件を見直してください。"
    )
  else:
    # パターン1: 間隔尺度（数値型データ） -> 平均値比較 & t検定
    if is_verify_numeric:
      st.markdown(f"#### 📊 平均値比較: {q_title}（間隔尺度）")
      s_num = pd.to_numeric(df_active[verify_q], errors="coerce")
      vals_A = s_num.loc[idx_A].dropna()
      vals_B = s_num.loc[idx_B].dropna()

      if len(vals_A) > 0 and len(vals_B) > 0:
        res_df = pd.DataFrame(
            {
                "ターゲット A": [vals_A.mean(), vals_A.std(), len(vals_A)],
                "ターゲット B": [vals_B.mean(), vals_B.std(), len(vals_B)],
            },
            index=["平均値", "標準偏差", "サンプルサイズ"],
        )
        st.dataframe(res_df.round(2))

        t_res = ttest_ind(vals_A, vals_B, equal_var=False)
        test_stat = float(t_res.statistic)
        p_val = float(t_res.pvalue)

        m1, m2, m3 = st.columns(3)
        m1.metric("t値", f"{test_stat:.2f}")
        m2.metric(
            "p値", f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}"
        )
        is_sig = p_val < 0.05
        is_sig_str = "有意差あり (p<0.05)" if is_sig else "有意差なし"
        m3.metric(
            "検定結果 (t検定)", is_sig_str, delta="有意" if is_sig else "なし"
        )

    # パターン2: カテゴリ尺度 -> クロス集計 & カイ二乗検定
    else:
      st.markdown(f"#### 📊 クロス集計結果: {q_title}（ターゲットA vs ターゲットB）")
      series_A = pd.DataFrame(
          {"グループ": "ターゲット A", "回答コード": df_active.loc[idx_A, verify_q]}
      )
      series_B = pd.DataFrame(
          {"グループ": "ターゲット B", "回答コード": df_active.loc[idx_B, verify_q]}
      )
      comp_df = pd.concat([series_A, series_B]).dropna()
      comp_df = comp_df[comp_df["回答コード"].isin(choices.keys())]

      if focus_choices:
        labels = [choices.get(c, c) for c in focus_choices]
        target_lbl = " / ".join(labels)
        comp_df["回答"] = np.where(
            comp_df["回答コード"].isin(focus_choices),
            target_lbl,
            f"その他（{target_lbl}以外）",
        )
      else:
        comp_df["回答"] = comp_df["回答コード"].map(choices)

      if len(comp_df["回答"].unique()) > 1:
        ct_pct = (
            pd.crosstab(comp_df["グループ"], comp_df["回答"], normalize="index")
            * 100
        )
        st.dataframe(ct_pct.round(1))

        ct_raw = pd.crosstab(comp_df["グループ"], comp_df["回答"])
        try:
          chi2, p_val, dof, _ = chi2_contingency(ct_raw)
          test_stat = float(chi2)
          m1, m2, m3 = st.columns(3)
          m1.metric("カイ二乗値 (χ²)", f"{test_stat:.2f}")
          m2.metric(
              "p値", f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}"
          )
          is_sig = p_val < 0.05
          is_sig_str = "有意差あり (p<0.05)" if is_sig else "有意差なし"
          m3.metric(
              "検定結果 (カイ二乗)",
              is_sig_str,
              delta="有意" if is_sig else "なし",
          )
        except Exception:
          st.info("※度数分布の偏りによりカイ二乗検定を実行できませんでした。")

# --- 7. スプレッドシート保存 ---
st.divider()
st.subheader("📋 3. 班のスプレッドシートに保存")
GAS_URL = "https://script.google.com/macros/s/AKfycby2qTjYVCz99zAtGx19DF4M1wkm0MTKoM417CpUpBPRytS18vSQpILcliYbFmm7-2Vugg/exec"

desc_A = " ＆ ".join(summary_A) if summary_A else "全数"
desc_B = " ＆ ".join(summary_B) if summary_B else "全数"

record_q = q_title
if focus_choices:
  labels = [choices.get(c, c) for c in focus_choices]
  record_q = f"{q_title} [注目: {' / '.join(labels)}]"

if st.button("この比較結果をスプレッドシートに追記する"):
  if "ここにGAS" in GAS_URL:
    st.warning("⚠️ まだGASのWebアプリURLが設定されていません。")
  else:
    payload = {
        "user_name": st.session_state.get("user_name", "メンバー"),
        "target_name": f"[A] {desc_A} vs [B] {desc_B}",
        "sample_size": f"A:{nA} / B:{nB}",
        "total_ratio": (
            f"A:{(nA/total_n)*100:.1f}% / B:{(nB/total_n)*100:.1f}%"
        ),
        "question_title": record_q,
        "chi2": round(float(test_stat), 2),
        "p_value": round(float(p_val), 4),
        "is_significant": is_sig_str,
    }
    try:
      res = requests.post(GAS_URL, json=payload, timeout=5)
      if res.status_code == 200:
        st.success("🎉 スプレッドシートに比較結果が記録されました！")
      else:
        st.error(f"送信エラー: {res.status_code}")
    except Exception as e:
      st.error(f"通信エラー: {e}")