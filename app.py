import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("📸 Demo xử lý ảnh nâng cao")
st.write("Chọn loại xử lý ảnh:")

option = st.selectbox(
    "Chọn xử lý",
    ["🚗 Tiền xử lý biển số xe", "🛰️ Cải thiện ảnh vệ tinh GIS", "🌙 Ảnh chụp ánh sáng kém"]
)

uploaded_file = st.file_uploader("Tải ảnh lên", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    st.image(img_array, caption="Ảnh gốc", use_column_width=True)

    # --- 1. Tiền xử lý biển số xe ---
    if option == "🚗 Tiền xử lý biển số xe":
        # Giữ màu nhưng thêm bước grayscale riêng cho cạnh
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_clahe = clahe.apply(gray)

        # Lọc nhiễu
        filtered = cv2.bilateralFilter(gray_clahe, 9, 75, 75)

        # Canny edge
        edges = cv2.Canny(filtered, 100, 200)

        st.image([gray, gray_clahe, filtered, edges],
                 caption=["Grayscale", "CLAHE", "Bilateral Filter", "Canny Edge"],
                 use_column_width=True)

    # --- 2. Cải thiện ảnh vệ tinh GIS ---
    elif option == "🛰️ Cải thiện ảnh vệ tinh GIS":
        # Cân bằng histogram trên kênh Y (YCrCb) => giữ màu
        ycrcb = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y_eq = cv2.equalizeHist(y)
        ycrcb_eq = cv2.merge([y_eq, cr, cb])
        img_eq = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2RGB)

        # Unsharp Masking (làm nét)
        gaussian = cv2.GaussianBlur(img_array, (9, 9), 10.0)
        unsharp = cv2.addWeighted(img_array, 1.5, gaussian, -0.5, 0)

        # Bilateral filter (làm mượt nhưng giữ cạnh)
        bilateral = cv2.bilateralFilter(img_array, 9, 75, 75)

        st.image([img_eq, unsharp, bilateral],
                 caption=["Histogram Equalization", "Unsharp Masking", "Bilateral Filter"],
                 use_column_width=True)

    # --- 3. Nâng cao ảnh ánh sáng kém ---
    elif option == "🌙 Ảnh chụp ánh sáng kém":
        # Chuyển sang YCrCb và tăng sáng trên kênh Y
        ycrcb = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        # CLAHE trên kênh Y (giữ màu gốc)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        y_clahe = clahe.apply(y)
        ycrcb_clahe = cv2.merge([y_clahe, cr, cb])
        img_clahe = cv2.cvtColor(ycrcb_clahe, cv2.COLOR_YCrCb2RGB)

        # Giảm nhiễu Non-local Means
        denoised = cv2.fastNlMeansDenoisingColored(img_clahe, None, 10, 10, 7, 21)

        st.image([img_clahe, denoised],
                 caption=["CLAHE trên kênh Y", "Sau giảm nhiễu"],
                 use_column_width=True)
