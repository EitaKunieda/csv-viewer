import streamlit as st
import pandas as pd
import dropbox

# --- ログイン機能 ---
def check_login():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        username = st.text_input("ユーザー名")
        password = st.text_input("パスワード", type="password")
        if st.button("ログイン"):
            users = st.secrets["users"]
            if username in users and users[username] == password:
                st.session_state["logged_in"] = True
            else:
                st.error("ユーザー名またはパスワードが違います")
                st.stop()
    if not st.session_state["logged_in"]:
        st.stop()

check_login()

st.title("Dropbox CSV Viewer")

# --- DropboxからCSVを取得 ---
def get_csv_from_dropbox(path="/在庫データ/在庫データ.csv"):
    dbx = dropbox.Dropbox(st.secrets["DROPBOX_TOKEN"])
    _, res = dbx.files_download(path)
    return pd.read_csv(res.raw)

df = get_csv_from_dropbox()

# 表表示（フィルタ・検索可能）
st.dataframe(df)

# 簡易グラフ
st.bar_chart(df.set_index(df.columns[0]))
