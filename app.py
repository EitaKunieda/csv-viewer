# app.py
# ---------------------------------------------
# カメラでJAN（EAN-13）を読み取り、太り/欠け補正度をスライダーで調整できる
# * 補正度0 = レーザースキャナ風（補正なし・規格どおりで厳しめ）
# * 補正度↑ = 形態学的処理（開閉処理）で太り/欠けを緩和して読取りやすくする
#
# 事前インストール:
#   pip install streamlit streamlit-webrtc opencv-python zxing-cpp numpy
# 実行:
#   streamlit run app.py
# ---------------------------------------------

import time
import queue
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase

# ZXing-C++ Python バインディング
try:
    import zxingcpp  # 高速・高精度
except Exception as e:
    zxingcpp = None

st.set_page_config(
    page_title="JAN バーコードスキャナ（補正度調整付き）",
    page_icon="📷",
    layout="wide",
)

# STUN（NAT越え用）
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# -------------------------------
# UI：設定
# -------------------------------
col_l, col_r = st.columns([1, 1])

with col_l:
    st.title("📷 JANコード スキャナ")
    st.caption("PC内蔵/スマホカメラに対応（ブラウザのカメラ許可が必要）")

    strict_mode = st.toggle("レーザースキャナ風の厳密読み取り（補正なし）", value=False,
                            help="オンにすると補正度は 0（無補正）になり、規格外の太り/欠けに弱くなります。")

    comp_level = st.slider(
        "太り/欠けの補正度",
        min_value=0, max_value=10, value=4, step=1,
        help="0=補正なし（厳格） / 値が大きいほど太りや欠けに強くなります（形態学的開閉処理）。",
        disabled=strict_mode,
    )

    # 背面カメラ利用（スマホ想定）
    use_back_camera = st.toggle("スマホ等で背面カメラを優先（facingMode='environment'）", value=True)

with col_r:
    st.subheader("読み取りガイド")
    st.markdown(
        "- できるだけフレームの中央にバーコードを合わせ、平行にしてください。\n"
        "- 照明ムラや反射を避け、ピントが合う距離に調整してください。\n"
        "- **補正度0** ＝ レーザースキャナ的に**補正しない**（規格外は読み取りにくい）。\n"
        "- **補正度↑** ＝ 太り/欠けを**やや補正**して読みやすくします。"
    )

# -------------------------------
# 読み取り結果の保持
# -------------------------------
if "scans" not in st.session_state:
    st.session_state.scans = []  # List[dict]: {"code": str, "format": str, "jan": bool, "time": float}

# 重複計上を抑制（同一コードを短時間に多重登録しない）
DEDUP_WINDOW_SEC = 2.0
if "recent_seen" not in st.session_state:
    st.session_state.recent_seen = {}  # code -> last_time

# -------------------------------
# 画像前処理（補正度に応じて）
# -------------------------------
def preprocess_for_barcode(frame_bgr: np.ndarray, level: int) -> np.ndarray:
    """
    level=0: 無補正（グレースケール＋2値化のみ）
    level>=1: 形態学的 開閉処理（太り/欠けの緩和）
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # コントラスト軽め強調＋2値化
    gray = cv2.equalizeHist(gray)
    # 自動閾値（Otsu）
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if level <= 0:
        return bw

    # levelに応じて構造要素を拡大
    k = 1 + level  # 1～11
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))   # 横方向（縦縞の太り補正）
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))  # 縦方向（欠け補正）

    # まず「開処理」（太り＝膨張によるバーの肥大を削る）
    opened = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    # 次に「閉処理」（欠け＝バー内のギャップを埋める）
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel2, iterations=1)
    return closed

# -------------------------------
# ZXingでデコード
# -------------------------------
def decode_barcodes(img: np.ndarray):
    """
    imgは8bit単一チャンネル or BGR を想定。
    戻り値: List[(text, format)]
    """
    results = []
    if zxingcpp is None:
        return results

    try:
        # ZXingはBGRでもグレースケールでもOK
        symbols = zxingcpp.read_barcodes(img)
        for sym in symbols:
            text = sym.text
            fmt = str(sym.format)  # zxingcpp.BarcodeFormat
            results.append((text, fmt))
    except Exception:
        pass
    return results

# -------------------------------
# JAN判定（EAN-13のうち日本GS1プレフィックス）
# -------------------------------
def is_jan_ean13(code: str) -> bool:
    """
    JANはEANN-13の日本割当（先頭'45'または'49'）が一般的。
    書籍などは 978/979 なども実運用されるが、ここでは 45/49 をJANとして扱う。
    """
    return len(code) == 13 and code.isdigit() and code.startswith(("45", "49"))

# -------------------------------
# VideoProcessor（webrtc）
# -------------------------------
@dataclass
class ProcessorState:
    comp_level: int = 0
    strict_mode: bool = False

class BarcodeProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.state = ProcessorState()
        self.last_frame_ts = 0.0

    def recv(self, frame):
        img_bgr = frame.to_ndarray(format="bgr24")

        # 有効な補正度
        level = 0 if self.state.strict_mode else self.state.comp_level

        # 前処理（レーザースキャナ風＝level=0なら無補正）
        proc = preprocess_for_barcode(img_bgr, level)

        # ZXingで読み取り
        decoded = decode_barcodes(proc)

        # 検出枠などを可視化するために元フレームへ重ね描き
        vis = img_bgr.copy()
        now = time.time()
        for text, fmt in decoded:
            # フォーマット名の整形
            fmt_name = fmt.replace("BarcodeFormat.", "")

            # 表示テキスト
            tag = f"{fmt_name}: {text}"
            color = (0, 180, 0) if is_jan_ean13(text) else (0, 0, 200)
            cv2.rectangle(vis, (10, 10), (int(10 + 9*len(tag)), 55), (255, 255, 255), -1)
            cv2.putText(vis, tag, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

            # 結果をSessionStateに格納（デバウンス）
            last = st.session_state.recent_seen.get(text, 0)
            if now - last > DEDUP_WINDOW_SEC:
                st.session_state.scans.append({
                    "コード": text,
                    "形式": fmt_name,
                    "JAN(45/49)": "Yes" if is_jan_ean13(text) else "No",
                    "時刻": time.strftime("%H:%M:%S"),
                    "補正度": level,
                    "厳密": "Yes" if self.state.strict_mode else "No",
                })
                st.session_state.recent_seen[text] = now

        return av.VideoFrame.from_ndarray(vis, format="bgr24")

# PyAVの遅延インポート（ストリーム開始時に使用）
import av  # noqa: E402

# -------------------------------
# WebRTC ストリーマ
# -------------------------------
media_constraints = {
    "video": {
        "facingMode": {"ideal": "environment" if use_back_camera else "user"},
        # リーダーの安定性を上げるために解像度を少し上げる
        "width": {"ideal": 1280},
        "height": {"ideal": 720},
        "frameRate": {"ideal": 30},
    },
    "audio": False,
}

webrtc_ctx = webrtc_streamer(
    key="jan-scanner",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints=media_constraints,
    video_processor_factory=BarcodeProcessor,
    async_processing=True,
)

# スライダー状態を VideoProcessor に反映
if webrtc_ctx and webrtc_ctx.video_processor:
    webrtc_ctx.video_processor.state.comp_level = 0 if strict_mode else int(comp_level)
    webrtc_ctx.video_processor.state.strict_mode = bool(strict_mode)

# -------------------------------
# 読み取り結果表示
# -------------------------------
st.divider()
st.subheader("📋 読み取り履歴")

if len(st.session_state.scans) == 0:
    st.info("まだ読み取り結果がありません。カメラにバーコード（JAN/EAN-13）をかざしてください。")
else:
    df = pd.DataFrame(st.session_state.scans)
    # 新しい順に表示
    df = df.iloc[::-1].reset_index(drop=True)
    st.dataframe(df, use_container_width=True)

# クリアボタン
cols = st.columns([1, 1, 5])
with cols[0]:
    if st.button("履歴をクリア"):
        st.session_state.scans = []
        st.session_state.recent_seen = {}
with cols[1]:
    st.download_button(
        "CSVで保存",
        data=pd.DataFrame(st.session_state.scans).to_csv(index=False).encode("utf-8-sig"),
        file_name="jan_scans.csv",
        mime="text/csv",
    )

# -------------------------------
# 参考メモ（サイドバー）
# -------------------------------
with st.sidebar:
    st.header("設定サマリ")
    st.write(
        f"- 厳密モード: {'ON' if strict_mode else 'OFF'}\n"
        f"- 補正度: {0 if strict_mode else comp_level}\n"
        f"- カメラ: {'背面優先' if use_back_camera else '前面/内蔵'}\n"
    )
    st.caption(
        "※ 補正度は開閉処理（Open/Close）で太り/欠けを緩和します。"
        "0は補正無し＝レーザースキャナ的な厳格モードです。"
    )













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






