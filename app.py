import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Demo Xử Lý Ảnh", layout="wide")

st.title("📷 Demo Xử Lý Ảnh với OpenCV & Streamlit")

uploaded_file = st.file_uploader("📂 Tải lên một ảnh để xử lý", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    st.image(image, caption="Ảnh gốc", use_container_width=True)

    option = st.sidebar.radio(
        "Chọn loại xử lý ảnh",
        (
            "🚗 Tiền xử lý cho nhận dạng biển số xe",
            "🛰️ Cải thiện ảnh vệ tinh trong GIS",
            "🌙 Nâng cao chất lượng ảnh ánh sáng kém",
        )
    )

    # 🚗 1. Tiền xử lý cho nhận dạng biển số xe
    if option == "🚗 Tiền xử lý cho nhận dạng biển số xe":
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        bilateral = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
        edges = cv2.Canny(bilateral, 100, 200)

        st.image(gray, caption="Ảnh xám (Grayscale)", use_container_width=True, channels="GRAY")
        st.image(enhanced, caption="Tăng tương phản (CLAHE)", use_container_width=True, channels="GRAY")
        st.image(bilateral, caption="Lọc nhiễu (Bilateral Filter)", use_container_width=True, channels="GRAY")
        st.image(edges, caption="Làm nổi bật biên cạnh (Canny)", use_container_width=True, channels="GRAY")

    # 🛰️ 2. Cải thiện ảnh vệ tinh trong GIS
    elif option == "🛰️ Cải thiện ảnh vệ tinh trong GIS":
        img_yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        hist_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

        gaussian_blur = cv2.GaussianBlur(hist_eq, (9, 9), 10.0)
        unsharp = cv2.addWeighted(hist_eq, 1.5, gaussian_blur, -0.5, 0)

        bilateral = cv2.bilateralFilter(unsharp, d=9, sigmaColor=75, sigmaSpace=75)

        st.image(hist_eq, caption="Cân bằng Histogram", use_container_width=True)
        st.image(unsharp, caption="Tăng độ sắc nét (Unsharp Masking)", use_container_width=True)
        st.image(bilateral, caption="Làm mượt nhưng giữ chi tiết (Bilateral Filter)", use_container_width=True)

    # 🌙 3. Nâng cao chất lượng ảnh ánh sáng kém
    elif option == "🌙 Nâng cao chất lượng ảnh ánh sáng kém":
        img_ycrcb = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(img_ycrcb)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        y_clahe = clahe.apply(y)

        merged = cv2.merge([y_clahe, cr, cb])
        enhanced = cv2.cvtColor(merged, cv2.COLOR_YCrCb2RGB)

        denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)

        st.image(enhanced, caption="Cải thiện độ sáng với CLAHE", use_container_width=True)
        st.image(denoised, caption="Giảm nhiễu (Non-local Means Denoising)", use_container_width=True)
