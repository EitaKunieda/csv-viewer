import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
from aspose.barcode.barcoderecognition import BarCodeReader

st.title("バーコード撮影＆読み取り（枠付き）")

# ガイド枠付きのイメージを作成（透明背景に赤枠）
def create_guide_overlay(width=640, height=480):
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    margin = 80
    draw.rectangle(
        [margin, margin, width - margin, height - margin],
        outline=(255, 0, 0, 200),
        width=5,
    )
    return img

st.markdown("📸 バーコードを赤い枠の中に合わせて撮影してください")

# ガイド枠を表示
guide = create_guide_overlay()
st.image(guide, caption="ガイド枠（参考用）", use_column_width=True)

# カメラ入力
camera_file = st.camera_input("バーコードを撮影")

# 補正スライダー
correction = st.slider("太り・欠け補正度", -4.0, 2.0, 0.0, 0.1)

if camera_file is not None:
    # 撮影画像を読み込み
    image = Image.open(camera_file).convert("RGB")
    img_array = np.array(image)

    # 画像処理（膨張 or 収縮）
    if correction != 0:
        ksize = max(1, int(round(abs(correction) * 3)))
        kernel = np.ones((ksize, ksize), np.uint8)
        if correction > 0:
            img_array = cv2.dilate(img_array, kernel, iterations=1)
        else:
            img_array = cv2.erode(img_array, kernel, iterations=1)

    # 前処理後の画像を保存
    tmp_path = "tmp_camera_corrected.png"
    cv2.imwrite(tmp_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

    st.image(img_array, caption=f"補正後画像（補正度={correction:.1f}）", use_column_width=True)

    # Aspose.Barcodeで読み取り
    reader = BarCodeReader(tmp_path)
    results = reader.read_bar_codes()

    if results:
        st.subheader("読み取り結果")
        for result in results:
            st.write(f"**タイプ**: {result.code_type_name}")
            st.write(f"**データ**: {result.code_text}")
    else:
        st.error("バーコードを読み取れませんでした。枠に正しく合わせて再試行してください。")


if 0:
    
    import streamlit as st
    import cv2
    import numpy as np
    from PIL import Image
    from aspose.barcode.barcoderecognition import BarCodeReader
    
    st.title("バーコード画像アップロード＆読み取り（受入試験用）")
    
    uploaded_file = st.file_uploader("バーコード画像をアップロードしてください", type=["png", "jpg", "jpeg"])
    
    # スライダーを小数で設定
    correction = st.slider("太り・欠け補正度", -4.0, 2.0, 0.0, 0.1)
    
    if uploaded_file is not None:
        # 画像を読み込み
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
    
        # OpenCVで前処理（膨張 or 収縮）
        if correction != 0:
            # スライダーの絶対値に応じてカーネルサイズを連続的に変化
            ksize = max(1, int(round(abs(correction) * 3)))
            kernel = np.ones((ksize, ksize), np.uint8)
    
            if correction > 0:
                img_array = cv2.dilate(img_array, kernel, iterations=1)
            else:
                img_array = cv2.erode(img_array, kernel, iterations=1)
    
        # 前処理後の画像を保存
        tmp_path = "tmp_corrected.png"
        cv2.imwrite(tmp_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    
        st.image(img_array, caption=f"補正後画像（補正度={correction:.1f}）", use_column_width=True)
    
        # Aspose.Barcodeで読み取り
        reader = BarCodeReader(tmp_path)
        results = reader.read_bar_codes()
    
        if results:
            st.subheader("読み取り結果")
            for result in results:
                st.write(f"**タイプ**: {result.code_type_name}")
                st.write(f"**データ**: {result.code_text}")
        else:
            st.error("バーコードを読み取れませんでした。補正度を変えて再試行してください。")
    


if 0:
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
    def get_csv_from_dropbox(path="/ZAIKO.csv"):
        print("getCSVの開始")
        dbx = dropbox.Dropbox(
            oauth2_refresh_token=st.secrets["REFRESH_TOKEN"],
            app_key=st.secrets["CLIENT_ID"],
            app_secret=st.secrets["CLIENT_SECRET"]
        )
        print("ああああ")
        
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
            df = get_csv_from_dropbox("/ZAIKO.csv")
        except Exception as e:
            st.error(f"DropboxからCSVを取得できませんでしたver2: {e}")
            return
    
        # 商品CD入力
        product_code = st.text_input("商品CDを入力")
    
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






