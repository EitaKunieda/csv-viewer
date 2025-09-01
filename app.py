
import streamlit as st
import pandas as pd
import dropbox
import io
import altair as alt

# -----------------------------
# 社員ごとのログイン認証
# -----------------------------
def login():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        st.title("社員ログイン")

        username = st.text_input("ユーザー名")
        password = st.text_input("パスワード", type="password")

        if st.button("ログイン"):
            users = st.secrets["users"]
            if username in users and password == users[username]:
                st.session_state["authenticated"] = True
                st.success("ログイン成功")
            else:
                st.error("ユーザー名またはパスワードが間違っています")
        st.stop()

# -----------------------------
# Dropbox から CSV を取得
# -----------------------------
@st.cache_data
def get_csv_from_dropbox(path="/data.csv"):
    dbx = dropbox.Dropbox(
        oauth2_refresh_token=st.secrets["REFRESH_TOKEN"],
        app_key=st.secrets["CLIENT_ID"],
        app_secret=st.secrets["CLIENT_SECRET"]
    )
    _, res = dbx.files_download(path)
    return pd.read_csv(io.BytesIO(res.content))

# -----------------------------
# メイン処理
# -----------------------------
def main():
    login()

    st.title("社内CSVビューア")

    # CSV読込
    try:
        df = get_csv_from_dropbox("/在庫データ/在庫データ.csv")
    except Exception as e:
        st.error(f"DropboxからCSVを取得できませんでした: {e}")
        return

    # 検索
    query = st.text_input("検索キーワードを入力")
    filtered_df = df
    if query:
        filtered_df = df[df.astype(str).apply(lambda row: row.str.contains(query, case=False, na=False)).any(axis=1)]

    # データ表示
    st.dataframe(filtered_df, use_container_width=True)

    # グラフ表示
    if not filtered_df.empty:
        numeric_cols = filtered_df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            col = st.selectbox("グラフ化する列を選択", numeric_cols)
            chart = (
                alt.Chart(filtered_df)
                .mark_bar()
                .encode(x=col, y="count()")
                .properties(width=600)
            )
            st.altair_chart(chart, use_container_width=True)

if __name__ == "__main__":
    main()

