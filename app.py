import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Xử lý ảnh nâng cao 🎨")

uploaded_file = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc ảnh
    image = Image.open(uploaded_file)
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Hiển thị ảnh gốc
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Ảnh gốc", use_container_width=True)

    st.subheader("Chọn nhóm xử lý ảnh")

    group = st.selectbox(
        "Chọn nhóm:",
        ("Nhận dạng biển số xe", "Ảnh vệ tinh GIS", "Ảnh chụp ánh sáng kém")
    )

    option = None
    if group == "Nhận dạng biển số xe":
        option = st.selectbox("Chọn kỹ thuật:",
            ("Grayscale", "Gaussian Blur", "CLAHE", "Canny Edge Detection")
        )

        if option == "Grayscale":
            result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif option == "Gaussian Blur":
            result = cv2.GaussianBlur(img, (5,5), 0)
        elif option == "CLAHE":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            result = clahe.apply(gray)
        elif option == "Canny Edge Detection":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            result = cv2.Canny(gray, 100, 200)

    elif group == "Ảnh vệ tinh GIS":
        option = st.selectbox("Chọn kỹ thuật:",
            ("Sharpening", "Histogram Equalization", "Median Filter")
        )

        if option == "Sharpening":
            kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
            result = cv2.filter2D(img, -1, kernel)
        elif option == "Histogram Equalization":
            yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        elif option == "Median Filter":
            result = cv2.medianBlur(img, 5)

    elif group == "Ảnh chụp ánh sáng kém":
        option = st.selectbox("Chọn kỹ thuật:",
            ("Gamma Correction", "CLAHE (Color)", "Denoising")
        )

        if option == "Gamma Correction":
            gamma = st.slider("Chọn gamma", 0.1, 3.0, 1.5)
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0,256)]).astype("uint8")
            result = cv2.LUT(img, table)
        elif option == "CLAHE (Color)":
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            result = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        elif option == "Denoising":
            result = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    # Hiển thị kết quả
    if len(result.shape) == 2:  # ảnh grayscale
        st.image(result, caption=f"Kết quả: {group} - {option}", use_container_width=True)
    else:
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption=f"Kết quả: {group} - {option}", use_container_width=True)
