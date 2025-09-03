import streamlit as st
import cv2
import numpy as np
from PIL import Image
from aspose.barcode.barcoderecognition import BarCodeReader

st.title("バーコード画像アップロード＆読み取り（受入試験用）")

uploaded_file = st.file_uploader("バーコード画像をアップロードしてください", type=["png", "jpg", "jpeg"])

# 補正度は「何モジュール太らせ/痩せさせるか」を表す
correction_modules = st.slider("太り・欠け補正（単位：モジュール）", -2.0, 2.0, 0.0, 0.1)

def rotate90_if_needed(gray: np.ndarray):
    # ざっくり：列方向の濃度変動(=バーが縦のとき大きい)と、行方向の濃度変動を比べる
    cols_std = gray.mean(axis=0).std()
    rows_std = gray.mean(axis=1).std()
    if rows_std > cols_std:
        return cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE), 90
    else:
        return gray, 0

def estimate_module_px(rot_gray: np.ndarray):
    # コントラスト標準化＋2値化
    blur = cv2.GaussianBlur(rot_gray, (5,5), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 列方向に平均して1次元化（縦バーを仮定）
    black_ratio = (thr == 0).mean(axis=0)  # 0.0〜1.0
    # 1Dを再2値化して黒(=1)/白(=0)のランレングスを測る
    line = (black_ratio > 0.5).astype(np.uint8)
    if line.sum() == 0:
        return None

    diff = np.diff(np.r_[0, line, 0])
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]
    lengths = ends - starts  # 黒バーの横幅(ピクセル)のリスト

    if len(lengths) == 0:
        return None

    # 最小バー幅に近い値として10パーセンタイルを採用（外れ値に強い）
    module_px = int(max(1, round(np.percentile(lengths, 10))))
    return module_px

def width_correct_by_modules(img_gray: np.ndarray, corr_modules: float):
    # バーの向きを縦に正規化（縦バー前提で横方向に補正を掛ける）
    rot_gray, rot_angle = rotate90_if_needed(img_gray)

    module_px = estimate_module_px(rot_gray)
    if module_px is None or corr_modules == 0.0:
        # 推定できないときは無加工を返す
        return img_gray

    # 「何モジュール」 → ピクセルに換算
    k = int(round(abs(corr_modules) * module_px))
    if k < 1:
        return img_gray

    # 横方向だけに効くカーネル（1行×k列）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
    if corr_modules > 0:
        proc = cv2.dilate(rot_gray, kernel, iterations=1)   # 太らせる
    else:
        proc = cv2.erode(rot_gray,  kernel, iterations=1)   # 痩せさせる

    # 元の向きに戻す
    if rot_angle == 90:
        proc = cv2.rotate(proc, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return proc

if uploaded_file is not None:
    # 画像読み込み
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # グレースケールに
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # ★スケール・方向正規化に基づく補正（横だけにかける）
    corrected_gray = width_correct_by_modules(gray, correction_modules)

    # 表示用にRGBへ
    corrected_rgb = cv2.cvtColor(corrected_gray, cv2.COLOR_GRAY2RGB)

    # 一時保存してAsposeで読む
    tmp_path = "tmp_corrected.png"
    cv2.imwrite(tmp_path, cv2.cvtColor(corrected_rgb, cv2.COLOR_RGB2BGR))

    st.image(corrected_rgb, caption=f"補正後画像（補正={correction_modules:.1f} モジュール）", use_column_width=True)

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






