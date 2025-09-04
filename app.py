import streamlit as st
import cv2
import numpy as np
from PIL import Image
from aspose.barcode.barcoderecognition import BarCodeReader

st.title("バーコード画像アップロード＆読み取り（受入試験用）")

# ✅ 正解（期待）バーコードの入力欄
expected_code = st.text_input("正しいバーコード番号を入力してください（判定に使用）", value="")

uploaded_file = st.file_uploader("バーコード画像をアップロードしてください", type=["png", "jpg", "jpeg"])

# スライダーを小数で設定
correction = st.slider("太り・欠け補正度", -4.0, 2.0, 0.0, 0.1)

def show_judgement(label: str, status: str):
    """判定表示（色・文字サイズを変更）"""
    styles = {
        "match": ("#00b050", 48),      # 緑・大
        "partial": ("#ff9900", 44),    # 橙・やや大
        "no_match": ("#d00000", 48),   # 赤・大
        "info": ("#666666", 18),       # グレー・小
    }
    color, size = styles.get(status, ("#333333", 24))
    st.markdown(
        f'<div style="color:{color}; font-size:{size}px; font-weight:800; margin:8px 0;">{label}</div>',
        unsafe_allow_html=True
    )

def judge_result(scanned_text: str, code_type_name: str, expected: str):
    """
    判定ロジック：
      - Code39 の場合：評価版でも省略されない想定 → 完全一致か不一致
      - それ以外：後半が "***" で省略される想定
           → 先頭の "***" までを前方一致判定
           → 前半が一致：部分一致、前半不一致：不一致
      - 例外的に "***" が含まれない他タイプは厳密一致（完全一致/不一致）で扱う
    """
    s = (scanned_text or "").strip()
    e = (expected or "").strip()

    if e == "":
        return ("期待値が未入力です。上の入力欄に正しいバーコード番号を入力してください。", "info")

    # Code39 の判定（名称のゆらぎも吸収）
    is_code39 = "code39" in (code_type_name or "").replace(" ", "").lower()
    if is_code39:
        return ("完全一致", "match") if s == e else ("不一致", "no_match")

    # 他タイプ：評価版で "***" により後半省略されるケースに対応
    if "***" in s:
        prefix = s.split("***", 1)[0]
        if prefix and e.startswith(prefix):
            return ("部分一致", "partial")
        else:
            return ("不一致", "no_match")

    # "***" が無いケース（例外的に省略されなかった等）は厳密比較
    if s == e:
        return ("完全一致", "match")
    # 片方がもう片方の前半に一致する場合は参考として部分一致扱い
    if s and (e.startswith(s) or s.startswith(e)):
        return ("部分一致", "partial")
    return ("不一致", "no_match")

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
        for idx, result in enumerate(results, start=1):
            code_type = result.code_type_name or ""
            code_text = result.code_text or ""

            st.markdown(f"**{idx}. タイプ**：{code_type}")
            st.markdown(f"**データ**：`{code_text}`")

            label, status = judge_result(code_text, code_type, expected_code)
            show_judgement(label, status)

            # 判定の根拠メモ（小さめ）
            if ("***" in (code_text or "")) and (status in ["partial", "no_match"]):
                st.caption("※ 評価版の仕様により後半が \"***\" で省略されるため、先頭部分のみで判定しています。")
    else:
        st.error("バーコードを読み取れませんでした。補正度を変えて再試行してください。")


if 0:
    
    import streamlit as st
    import cv2
    import numpy as np
    from PIL import Image
    from aspose.barcode.barcoderecognition import BarCodeReader, DecodeType
    
    st.title("アップロード画像からバーコード検出")
    
    uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # PILで読み込み→OpenCV形式に変換
        image = Image.open(uploaded_file).convert("RGB")
        frame = np.array(image)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
        # Asposeでバーコード検出
        reader = BarCodeReader(uploaded_file, DecodeType.ALL_SUPPORTED_TYPES)
        results = reader.read_bar_codes()
    
        if results:
            for result in results:
                # region.points から四隅の座標を取得
                pts = result.region.points  # list of Point(x, y)
                x_coords = [p.x for p in pts]
                y_coords = [p.y for p in pts]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
    
                # 赤い矩形で囲む
                cv2.rectangle(frame_bgr, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
    
                # バーコード種類と内容を文字で表示
                label = f"{result.code_type_name}: {result.code_text}"
                cv2.putText(frame_bgr, label, (x_min, max(y_min-10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            
            st.success("バーコード検出成功")
        else:
            st.warning("バーコードが検出できませんでした")
    
        # OpenCV(BGR)→RGBに変換して表示
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, use_container_width=True)

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






