# app.py
import json
import numpy as np
import pandas as pd
import requests
from scipy.stats import chi2_contingency, ttest_ind
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


st.title("🥤 ターゲット分析 & クロス集計")
st.caption(f"母集団全体 N = {total_n:,}人 ｜ ログイン中: {st.session_state.user_name}")

# --- 3. ターゲット設定（3段階ドリルダウン × 4条件） ---
st.subheader("🎯 1. ターゲット条件の設定（4変数）")
st.write(
    "大分類を選んでから設問を絞り込み、条件を指定してください（間隔尺度の場合は数値範囲スライダーが表示されます）。"
)


def render_condition_selector(idx):
  st.markdown(f"#### 【条件 {idx}】")
  c1, c2 = st.columns(2)
  with c1:
    selected_cat = st.selectbox(
        f"① 大分類を選択 (条件{idx})", options=categories, key=f"cat_{idx}"
    )

  available_qs = cat_map.get(selected_cat, [])
  if not available_qs:
    st.warning("この分類に含まれる有効な設問がありません。")
    return None, None

  with c2:
    selected_q = st.selectbox(
        f"② 質問を選択 (条件{idx})",
        options=available_qs,
        format_func=lambda q: f"{questions[q]['title']} ({q})",
        key=f"q_{idx}_{selected_cat}",
    )

  q_info = questions[selected_q]
  choices = q_info.get("choices", {})
  is_numeric = (
      q_info.get("type") == "NUM"
      or len(choices) == 0
      or selected_q in ["AAF2"]
  )

  # 間隔尺度（数値型）の場合：スライダーUI
  if is_numeric:
    # 簡易取得して最小値・最大値を確認（デフォルト例: 15〜79）
    val_range = st.slider(
        f"③ 数値範囲を指定（「{q_info['title']}」）",
        min_value=15,
        max_value=85,
        value=(20, 29),
        key=f"val_range_{idx}_{selected_q}",
    )
    return selected_q, ("NUMERIC", val_range)

  # 質的データ（選択肢型）の場合：マルチセレクトUI
  selected_vals = st.multiselect(
      f"③ 該当する回答を選択（「{q_info['title']}」）",
      options=list(choices.keys()),
      format_func=lambda c: choices.get(c, c),
      key=f"val_{idx}_{selected_q}",
  )
  return selected_q, ("CATEGORICAL", selected_vals)


active_conditions = []
for i in range(1, 5):
  q_code, cond_val = render_condition_selector(i)
  if q_code and cond_val:
    active_conditions.append((q_code, cond_val))

# --- 4. クロス集計・仮説検証 設問選択 ---
st.divider()
st.subheader("🔍 2. 検証する質問の選択")
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

# 注目する回答（絞り込み・2値化用）の選択UI
focus_choices = []
if verify_q:
  q_info = questions[verify_q]
  choices = q_info.get("choices", {})
  is_verify_numeric = (
      q_info.get("type") == "NUM"
      or len(choices) == 0
      or verify_q in ["AAF2"]
  )

  # カテゴリ設問の場合のみ、特定の回答に絞り込むUIを表示
  if not is_verify_numeric and len(choices) > 0:
    focus_choices = st.multiselect(
        "③ 注目する回答を選択（未選択なら全選択肢を一覧表示）",
        options=list(choices.keys()),
        format_func=lambda c: choices.get(c, c),
        help=(
            "特定の回答（例: 東京都）を選ぶと、「該当回答」vs「その他全体」の2値に集約してすっきりクロス集計します。"
        ),
        key=f"focus_{verify_q}",
    )

# --- 5. オンデマンド読み込みと集計実行 ---
cols_to_load = [cond[0] for cond in active_conditions]
if verify_q:
  cols_to_load.append(verify_q)

df_active = load_selected_columns(cols_to_load)

# ターゲット判定マスク作成
target_mask = pd.Series(True, index=df_active.index)
filter_summary = []

for q_code, cond_data in active_conditions:
  mode, val = cond_data
  if q_code in df_active.columns:
    if mode == "NUMERIC":
      num_series = pd.to_numeric(df_active[q_code], errors="coerce")
      target_mask &= (num_series >= val[0]) & (num_series <= val[1])
      filter_summary.append(
          f"{questions[q_code]['title']}: {val[0]}〜{val[1]}"
      )
    else:
      if val:
        target_mask &= df_active[q_code].astype(str).isin(val)
        choice_labels = [questions[q_code]["choices"].get(v, v) for v in val]
        filter_summary.append(
            f"{questions[q_code]['title']}: {', '.join(choice_labels)}"
        )

target_n = int(target_mask.sum())
target_ratio = (target_n / total_n) * 100 if total_n > 0 else 0

st.info(
    f"📊 **設定したターゲットの規模**\n\n"
    f"・対象人数: **{target_n:,} 人** ｜ 全体比率: **{target_ratio:.2f}%**"
)

# 検証分析（カテゴリ尺度ならクロス集計、間隔尺度なら平均値比較）
test_stat, p_val = 0.0, 1.0
q_title = ""
is_sig_str = "有意差なし"

if verify_q and verify_q in df_active.columns and target_n > 0:
  q_info = questions[verify_q]
  q_title = q_info["title"]
  choices = q_info.get("choices", {})
  is_verify_numeric = (
      q_info.get("type") == "NUM"
      or len(choices) == 0
      or verify_q in ["AAF2"]
  )

  # パターンA：間隔尺度（数値型データ）の場合 -> 平均値比較 & t検定
  if is_verify_numeric:
    st.write(f"**【平均値の比較: {q_title}（間隔尺度）】**")
    numeric_s = pd.to_numeric(df_active[verify_q], errors="coerce")
    t_vals = numeric_s[target_mask].dropna()
    other_vals = numeric_s[~target_mask].dropna()

    if len(t_vals) > 0 and len(other_vals) > 0:
      mean_df = pd.DataFrame(
          {
              "ターゲット層": [t_vals.mean(), t_vals.std(), len(t_vals)],
              "その他全体": [
                  other_vals.mean(),
                  other_vals.std(),
                  len(other_vals),
              ],
          },
          index=["平均値", "標準偏差", "サンプルサイズ"],
      )
      st.dataframe(mean_df.round(2))

      t_res = ttest_ind(t_vals, other_vals, equal_var=False)
      test_stat = float(t_res.statistic)
      p_val = float(t_res.pvalue)

      col1, col2, col3 = st.columns(3)
      col1.metric("t値", f"{test_stat:.2f}")
      col2.metric("p値", f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}")
      is_sig = p_val < 0.05
      is_sig_str = "有意差あり (p<0.05)" if is_sig else "有意差なし"
      col3.metric(
          "検定結果 (t検定)", is_sig_str, delta="有意" if is_sig else "なし"
      )

  # パターンB：カテゴリ尺度（質的データ）の場合 -> クロス集計 & カイ二乗検定
  else:
    sub_df = pd.DataFrame({
        "グループ": np.where(target_mask, "ターゲット", "その他全体"),
        "回答コード": df_active[verify_q].astype(str),
    }).dropna()
    sub_df = sub_df[sub_df["回答コード"].isin(choices.keys())]

    if len(sub_df) > 0:
      # ★ 注目する回答が指定された場合：2値化（該当項目 vs その他）
      if focus_choices:
        focus_labels = [choices.get(c, c) for c in focus_choices]
        target_label = " / ".join(focus_labels)
        other_label = f"その他（{target_label}以外）"

        sub_df["回答ラベル"] = np.where(
            sub_df["回答コード"].isin(focus_choices), target_label, other_label
        )
        display_title = f"{q_title}（注目: {target_label}）"
      else:
        # 未指定時は従来の全カテゴリ表示
        sub_df["回答ラベル"] = sub_df["回答コード"].map(choices)
        display_title = q_title

      if len(sub_df["回答ラベル"].unique()) > 1:
        ct = pd.crosstab(sub_df["グループ"], sub_df["回答ラベル"])
        ct_pct = (
            pd.crosstab(
                sub_df["グループ"], sub_df["回答ラベル"], normalize="index"
            )
            * 100
        )

        st.write(f"**【クロス集計結果: {display_title}（比率 %）】**")
        st.dataframe(ct_pct.round(1))

        try:
          chi2, p_val, dof, exp = chi2_contingency(ct)
          test_stat = float(chi2)
          col1, col2, col3 = st.columns(3)
          col1.metric("カイ二乗値 (χ²)", f"{test_stat:.2f}")
          col2.metric(
              "p値", f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}"
          )
          is_sig = p_val < 0.05
          is_sig_str = "有意差あり (p<0.05)" if is_sig else "有意差なし"
          col3.metric(
              "検定結果 (カイ二乗)",
              is_sig_str,
              delta="有意" if is_sig else "なし",
          )
        except Exception:
          st.info("※度数分布の偏りによりカイ二乗検定を実行できませんでした。")
      else:
        st.warning("集計に必要なバリエーションが不足しています。")
    else:
      st.warning("有効な回答データが不足しているため、クロス集計できません。")

# --- 6. スプレッドシート保存 ---
st.divider()
st.subheader("📋 3. 班のスプレッドシートに保存")
GAS_URL = "https://script.google.com/macros/s/AKfycby2qTjYVCz99zAtGx19DF4M1wkm0MTKoM417CpUpBPRytS18vSQpILcliYbFmm7-2Vugg/exec"

target_desc = " ＆ ".join(filter_summary) if filter_summary else "全数（条件なし）"

# 保存用の検証設問名（注目項目がある場合は追記）
record_q_title = q_title
if focus_choices:
  focus_labels = [choices.get(c, c) for c in focus_choices]
  record_q_title = f"{q_title} [注目: {' / '.join(focus_labels)}]"

if st.button("この分析結果をスプレッドシートに追記する"):
  if "ここにGAS" in GAS_URL:
    st.warning("⚠️ まだGASのWebアプリURLが設定されていません。")
  else:
    payload = {
        "user_name": st.session_state.get("user_name", "メンバー"),
        "target_name": target_desc,
        "sample_size": target_n,
        "total_ratio": f"{target_ratio:.2f}%",
        "question_title": record_q_title,
        "chi2": round(float(test_stat), 2),
        "p_value": round(float(p_val), 4),
        "is_significant": is_sig_str,
    }
    try:
      res = requests.post(GAS_URL, json=payload, timeout=5)
      if res.status_code == 200:
        st.success("🎉 スプレッドシートに記録されました！")
      else:
        st.error(f"送信エラー: {res.status_code}")
    except Exception as e:
      st.error(f"通信エラー: {e}")