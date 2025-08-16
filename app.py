import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Xử lý ảnh minh họa", layout="wide")

st.title("📷 Demo Xử Lý Ảnh với OpenCV + Streamlit")

option = st.sidebar.selectbox(
    "Chọn bài toán xử lý ảnh:",
    (
        "🚗 Tiền xử lý nhận dạng biển số xe",
        "🛰️ Cải thiện ảnh vệ tinh trong GIS",
        "🌙 Nâng cao chất lượng ảnh chụp ánh sáng kém",
    )
)

uploaded_file = st.file_uploader("Tải ảnh lên", type=["jpg", "png", "jpeg"])

def show_images(original, processed, caption1="Ảnh gốc", caption2="Ảnh sau xử lý"):
    col1, col2 = st.columns(2)
    with col1:
        st.image(original, caption=caption1, use_container_width=True)
    with col2:
        st.image(processed, caption=caption2, use_container_width=True)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # ======================================================
    # 🚗 1. Tiền xử lý cho nhận dạng biển số xe
    # ======================================================
    if option == "🚗 Tiền xử lý nhận dạng biển số xe":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Tăng tương phản bằng CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Lọc nhiễu
        denoised = cv2.bilateralFilter(enhanced, 11, 17, 17)

        # Làm nổi bật biên cạnh
        edges = cv2.Canny(denoised, 30, 200)

        show_images(gray, edges, "Grayscale", "Biên cạnh (Canny)")

    # ======================================================
    # 🛰️ 2. Cải thiện ảnh vệ tinh trong GIS
    # ======================================================
    elif option == "🛰️ Cải thiện ảnh vệ tinh trong GIS":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Cân bằng histogram
        equalized = cv2.equalizeHist(gray)

        # Unsharp Masking
        gaussian = cv2.GaussianBlur(equalized, (9, 9), 10.0)
        unsharp = cv2.addWeighted(equalized, 1.5, gaussian, -0.5, 0)

        # Làm mượt giữ chi tiết
        smooth = cv2.bilateralFilter(unsharp, 9, 75, 75)

        show_images(gray, smooth, "Grayscale", "Sau cải thiện GIS")

    # ======================================================
    # 🌙 3. Nâng cao ảnh chụp ánh sáng kém
    # ======================================================
    elif option == "🌙 Nâng cao chất lượng ảnh chụp ánh sáng kém":
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        # CLAHE cho kênh Y (độ sáng)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        y_clahe = clahe.apply(y)

        merged = cv2.merge((y_clahe, cr, cb))
        brightened = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

        # Giảm nhiễu
        denoised = cv2.fastNlMeansDenoisingColored(brightened, None, 10, 10, 7, 21)

        show_images(img, denoised, "Ảnh gốc", "Sau cải thiện ánh sáng kém")

else:
    st.info("⬆️ Hãy tải lên một ảnh để bắt đầu xử lý.")
