# app.py
import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
from scipy.stats import chi2_contingency

st.set_page_config(page_title="生活者データ分析ツール", layout="centered", page_icon="🥤")

# --- 1. 合言葉認証 ---
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        st.markdown("### 🔒 ゼミ分析ツール（班内限定）")
        user_name = st.text_input("あなたのお名前（記録用）", value="メンバー")
        pw = st.text_input("合言葉を入力してください", type="password")
        if st.button("ログイン"):
            if pw == "analysis":
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
    # ベースのインデックス・母集団数確認（KEY列のみ超軽量読み込み）
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

# 指定した列だけをディスクからオンデマンド取得する関数（超省メモリ）
def load_selected_columns(col_list):
    valid_cols = list(set([c for c in col_list if c and c in questions or c == "KEY"]))
    if not valid_cols:
        valid_cols = ["KEY"]
    return pd.read_parquet("app_data.parquet", columns=valid_cols)

st.title("🥤 ターゲット分析 & クロス集計")
st.caption(f"母集団全体 N = {total_n:,}人 ｜ ログイン中: {st.session_state.user_name}")

# --- 3. ターゲット設定（3段階ドリルダウン × 4条件） ---
st.subheader("🎯 1. ターゲット条件の設定（4変数）")
st.write("大分類を選んでから設問を絞り込み、該当する回答を選択してください。")

def render_condition_selector(idx):
    st.markdown(f"#### 【条件 {idx}】")
    c1, c2 = st.columns(2)
    with c1:
        selected_cat = st.selectbox(f"① 大分類を選択 (条件{idx})", options=categories, key=f"cat_{idx}")
    
    available_qs = cat_map.get(selected_cat, [])
    if not available_qs:
        st.warning("この分類に含まれる有効な設問がありません。")
        return None, []
    
    with c2:
        selected_q = st.selectbox(
            f"② 質問を選択 (条件{idx})",
            options=available_qs,
            format_func=lambda q: f"{questions[q]['title']} ({q})",
            key=f"q_{idx}_{selected_cat}"
        )
    
    q_info = questions[selected_q]
    choices = q_info.get("choices", {})
    if not choices:
        st.info("この質問には選択肢情報がありません。")
        return None, []
    
    selected_vals = st.multiselect(
        f"③ 該当する回答を選択（「{q_info['title']}」）",
        options=list(choices.keys()),
        format_func=lambda c: choices.get(c, c),
        key=f"val_{idx}_{selected_q}"
    )
    return selected_q, selected_vals

active_conditions = []
for i in range(1, 5):
    q_code, vals = render_condition_selector(i)
    if q_code and vals:
        active_conditions.append((q_code, vals))

# --- 4. クロス集計・仮説検証 設問選択 ---
st.divider()
st.subheader("🔍 2. 検証する質問とのクロス集計")
v_col1, v_col2 = st.columns(2)
with v_col1:
    verify_cat = st.selectbox("① 検証質問の大分類", options=categories, key="verify_cat")

verify_available_qs = cat_map.get(verify_cat, [])
with v_col2:
    verify_q = st.selectbox(
        "② 検証する質問",
        options=verify_available_qs,
        format_func=lambda q: f"{questions[q]['title']} ({q})",
        key=f"verify_q_{verify_cat}"
    )

# --- 5. 選択された列だけをオンデマンド読み込みして集計実行 ---
cols_to_load = [cond[0] for cond in active_conditions]
if verify_q:
    cols_to_load.append(verify_q)

df_active = load_selected_columns(cols_to_load)

# ターゲット判定マスクの作成
target_mask = pd.Series(True, index=df_active.index)
filter_summary = []

for q_code, vals in active_conditions:
    if q_code in df_active.columns:
        target_mask &= df_active[q_code].astype(str).isin(vals)
        choice_labels = [questions[q_code]["choices"].get(v, v) for v in vals]
        filter_summary.append(f"{questions[q_code]['title']}: {', '.join(choice_labels)}")

target_n = int(target_mask.sum())
target_ratio = (target_n / total_n) * 100 if total_n > 0 else 0

st.info(f"📊 **設定したターゲットの規模**\n\n・対象人数: **{target_n:,} 人** ｜ 全体比率: **{target_ratio:.2f}%**")

# クロス集計とカイ二乗検定
chi2, p = 0.0, 1.0
q_title = ""

if verify_q and verify_q in df_active.columns:
    q_info = questions[verify_q]
    q_title = q_info["title"]
    choices = q_info.get("choices", {})

    if len(choices) > 0 and target_n > 0:
        sub_df = pd.DataFrame({
            "グループ": np.where(target_mask, "ターゲット", "その他全体"),
            "回答": df_active[verify_q].astype(str)
        }).dropna()

        sub_df = sub_df[sub_df["回答"].isin(choices.keys())]

        if len(sub_df) > 0 and len(sub_df["回答"].unique()) > 1:
            sub_df["回答ラベル"] = sub_df["回答"].map(choices)
            ct = pd.crosstab(sub_df["グループ"], sub_df["回答ラベル"])
            ct_pct = pd.crosstab(sub_df["グループ"], sub_df["回答ラベル"], normalize="index") * 100

            st.write(f"**【クロス集計結果: {q_title}（比率 %）】**")
            st.dataframe(ct_pct.round(1))

            try:
                chi2, p, dof, expected = chi2_contingency(ct)
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("カイ二乗値 (χ²)", f"{chi2:.2f}")
                col_m2.metric("p値", f"{p:.4e}" if p < 0.0001 else f"{p:.4f}")
                col_m3.metric("有意差の有無", "有意差あり (p<0.05)" if p < 0.05 else "有意差なし", delta="有意" if p < 0.05 else "なし")
            except Exception:
                st.info("※度数分布の偏りによりカイ二乗検定を実行できませんでした。")
        else:
            st.warning("有効な回答データが不足しているため、クロス集計できません。")
    else:
        st.warning("ターゲット対象者が0人、または選択肢情報が存在しません。")

# --- 6. スプレッドシート保存 ---
st.divider()
st.subheader("📋 3. 班のスプレッドシートに保存")
GAS_URL = "https://script.google.com/macros/s/AKfycbxJQ3XH6Ij2XDXRr0vlgEhHqG1sPi_5p07tKjmVlIt41Yfnpv1mfVUumzeNU5zYKq7H2A/exec"

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
            "is_significant": "有意差あり(p<0.05)" if p < 0.05 else "有意差なし"
        }
        try:
            res = requests.post(GAS_URL, json=payload, timeout=5)
            if res.status_code == 200:
                st.success("🎉 スプレッドシートに記録されました！")
            else:
                st.error(f"送信エラー: {res.status_code}")
        except Exception as e:
            st.error(f"通信エラー: {e}")