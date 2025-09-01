import streamlit as st
import pandas as pd
import dropbox
import io


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
def get_csv_from_dropbox(path="/APP/ZAIKO/ZAIKO.csv"):
    dbx = dropbox.Dropbox(
        oauth2_refresh_token=st.secrets["REFRESH_TOKEN"],
        app_key=st.secrets["CLIENT_ID"],
        app_secret=st.secrets["CLIENT_SECRET"]
    )
    res = dbx.files_list_folder("").entries
    for f in res:
        print("とりまパスリスト")
        print(f.path_lower)
    
    _, res = dbx.files_download(path)
    # 商品CDを文字列で扱うため dtype={"商品CD": str}
    return pd.read_csv(io.BytesIO(res.content), dtype={"商品CD": str}, parse_dates=["作成日時"])

# -----------------------------
# メイン処理
# -----------------------------
def main():
    print("プログラム開始ですます")
    login()
    st.title("商品別 在庫集計ビューア")

    try:
        df = get_csv_from_dropbox("/APP/ZAIKO/ZAIKO.csv")
    except Exception as e:
        st.error(f"DropboxからCSVを取得できませんでしたver2: {e}")
        return

    # 商品CD入力
    product_code = st.text_input("商品CDを入力（8桁ゼロ埋め可）")

    if product_code:
        # 前ゼロを保持した形で検索
        filtered = df[df["商品CD"] == product_code]

        if filtered.empty:
            st.warning("該当する商品が見つかりませんでした")
        else:
            # 商品名称を取得
            product_name = filtered["商品名称"].iloc[0]
            st.subheader(f"商品CD: {product_code} | 商品名称: {product_name}")

            # センターごと在庫数集計
            summary = filtered.groupby("センター", as_index=False)["在庫数"].sum()

            st.dataframe(summary, use_container_width=True)

            # 棒グラフ表示
            st.bar_chart(summary.set_index("センター"))

if __name__ == "__main__":
    main()






