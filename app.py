import cv2
import numpy as np
import streamlit as st
from aspose.barcode.barcoderecognition import BarCodeReader, DecodeType
from PIL import Image
import tempfile

st.title("バーコード検出＆四角で囲む")

# 画像アップロード
uploaded_file = st.file_uploader("バーコード画像をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 一時ファイルに保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # OpenCVで画像読み込み
    frame = cv2.imread(tmp_path)

    # Asposeでバーコード検出
    reader = BarCodeReader(tmp_path, DecodeType.ALL_SUPPORTED_TYPES)
    results = reader.read_bar_codes()   # ✅ 正しいメソッド名

    if results:
        for result in results:
            # bounding box取得
            region = result.region
            rect = region.get_boundary()
            x, y, w, h = rect.x, rect.y, rect.width, rect.height

            # 四角を描画
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # ラベル（種類とデータ）を表示
            label = f"{result.code_type_name}: {result.code_text}"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # BGR→RGBに変換してStreamlitで表示
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption="検出されたバーコード", use_column_width=True)

        # 結果リストを表示
        st.subheader("検出結果")
        for result in results:
            st.write(f"**タイプ**: {result.code_type_name}")
            st.write(f"**データ**: {result.code_text}")

    else:
        st.error("バーコードを検出できませんでした。")



if 0:
    
    
    import streamlit as st
    import cv2
    import numpy as np
    from PIL import Image
    from aspose.barcode.barcoderecognition import BarCodeReader
    
    st.title("バーコード画像アップロード＆読み取り（台形補正付き）")
    
    uploaded_file = st.file_uploader("バーコード画像をアップロードしてください", type=["png", "jpg", "jpeg"])
    
    # 補正度は「何モジュール太らせ/痩せさせるか」を表す
    correction_modules = st.slider("太り・欠け補正（単位：モジュール）", -2.0, 2.0, 0.0, 0.1)
    
    
    # --- 台形補正用ユーティリティ ---
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # 左上
        rect[2] = pts[np.argmax(s)]  # 右下
    
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # 右上
        rect[3] = pts[np.argmax(diff)]  # 左下
        return rect
    
    def four_point_transform(image, pts):
        rect = order_points(pts)
        (tl, tr, br, bl) = rect
    
        # 幅と高さを計算
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))
    
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))
    
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
    
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped
    
    
    def detect_barcode_quad(gray):
        # エッジ検出＋輪郭抽出
        edged = cv2.Canny(gray, 50, 200)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        if not contours:
            return None
    
        # 大きい輪郭を順に調べる
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2)
    
        return None
    
    
    # --- モジュール幅推定 ---
    def estimate_module_px(rot_gray: np.ndarray):
        blur = cv2.GaussianBlur(rot_gray, (5, 5), 0)
        _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        black_ratio = (thr == 0).mean(axis=0)
        line = (black_ratio > 0.5).astype(np.uint8)
        if line.sum() == 0:
            return None
        diff = np.diff(np.r_[0, line, 0])
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        n = min(len(starts), len(ends))
        if n == 0:
            return None
        lengths = ends[:n] - starts[:n]
        if len(lengths) == 0:
            return None
        module_px = int(max(1, round(np.percentile(lengths, 10))))
        return module_px
    
    
    # --- 補正処理 ---
    def width_correct_by_modules(img_gray: np.ndarray, corr_modules: float):
        module_px = estimate_module_px(img_gray)
        if module_px is None or corr_modules == 0.0:
            return img_gray
        k = int(round(abs(corr_modules) * module_px))
        if k < 1:
            return img_gray
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
        if corr_modules > 0:
            proc = cv2.dilate(img_gray, kernel, iterations=1)
        else:
            proc = cv2.erode(img_gray, kernel, iterations=1)
        return proc
    
    
    # --- メイン処理 ---
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
        # 台形補正
        quad = detect_barcode_quad(gray)
        if quad is not None:
            warped = four_point_transform(img_array, quad)
            st.image(warped, caption="台形補正後", use_column_width=True)
            gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
        else:
            st.warning("台形補正できませんでした（バーコード領域が検出されませんでした）")
            warped = img_array
    
        # 幅補正
        corrected_gray = width_correct_by_modules(gray, correction_modules)
        corrected_rgb = cv2.cvtColor(corrected_gray, cv2.COLOR_GRAY2RGB)
    
        # 保存＆表示
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






