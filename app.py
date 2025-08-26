import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.title("📸 Demo xử lý ảnh")

# Upload ảnh
uploaded_file = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")  # ép về RGB
    img_array = np.array(image)

    st.image(img_array, caption="Ảnh gốc", use_container_width=True)

    option = st.radio(
        "Chọn loại xử lý ảnh",
        [
            "🚗 Tiền xử lý nhận dạng biển số xe",
            "🛰️ Cải thiện ảnh vệ tinh GIS",
            "🌙 Nâng cao chất lượng ảnh ánh sáng kém"
        ]
    )

    # 🚗 1. Tiền xử lý nhận dạng biển số xe
    if option == "🚗 Tiền xử lý nhận dạng biển số xe":
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        clip = st.slider("ClipLimit (CLAHE)", 1.0, 5.0, 2.0, 0.1)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)

        sigma_color = st.slider("Sigma Color (Bilateral)", 10, 150, 75)
        sigma_space = st.slider("Sigma Space (Bilateral)", 10, 150, 75)
        bilateral = cv2.bilateralFilter(contrast, d=9, sigmaColor=sigma_color, sigmaSpace=sigma_space)

        t1 = st.slider("Canny Threshold1", 50, 200, 100)
        t2 = st.slider("Canny Threshold2", 100, 300, 200)
        edges = cv2.Canny(bilateral, t1, t2)

        st.image(gray, caption="Ảnh Grayscale", use_container_width=True, channels="GRAY")
        st.image(contrast, caption="Tăng tương phản (CLAHE)", use_container_width=True, channels="GRAY")
        st.image(bilateral, caption="Lọc nhiễu (Bilateral Filter)", use_container_width=True, channels="GRAY")
        st.image(edges, caption="Làm nổi bật biên cạnh (Canny)", use_container_width=True, channels="GRAY")

    # 🛰️ 2. Cải thiện ảnh vệ tinh GIS
    elif option == "🛰️ Cải thiện ảnh vệ tinh GIS":
        ycrcb = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        y_eq = cv2.equalizeHist(y)
        img_eq = cv2.merge((y_eq, cr, cb))
        img_eq = cv2.cvtColor(img_eq, cv2.COLOR_YCrCb2RGB)

        gaussian = cv2.GaussianBlur(img_eq, (9, 9), 10.0)
        unsharp = cv2.addWeighted(img_eq, 1.5, gaussian, -0.5, 0)

        bilateral = cv2.bilateralFilter(unsharp, d=9, sigmaColor=75, sigmaSpace=75)

        st.image(img_eq, caption="Cân bằng histogram", use_container_width=True)
        st.image(unsharp, caption="Tăng độ sắc nét (Unsharp Masking)", use_container_width=True)
        st.image(bilateral, caption="Làm mượt nhưng giữ chi tiết (Bilateral Filter)", use_container_width=True)

    # 🌙 3. Nâng cao chất lượng ảnh ánh sáng kém
    elif option == "🌙 Nâng cao chất lượng ảnh ánh sáng kém":
        ycrcb = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        y = cv2.convertScaleAbs(y, alpha=1.2, beta=30)  # tăng sáng nhẹ

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y_clahe = clahe.apply(y)

        img_clahe = cv2.merge((y_clahe, cr, cb))
        img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_YCrCb2RGB)

        h = st.slider("Hạt nhiễu (fastNlMeansDenoising)", 5, 20, 10)
        denoised = cv2.fastNlMeansDenoisingColored(img_clahe, None, h, h, 7, 21)

        st.image(img_clahe, caption="Ảnh sau CLAHE", use_container_width=True)
        st.image(denoised, caption="Ảnh sau giảm nhiễu (Denoising)", use_container_width=True)
