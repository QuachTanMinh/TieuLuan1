import cv2
import numpy as np
import streamlit as st

st.title("🖼️ Ứng dụng Xử lý Ảnh Demo")
st.write("Chọn loại xử lý ảnh phù hợp với nhu cầu của bạn")

uploaded_file = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png"])

option = st.selectbox(
    "Chọn chế độ xử lý:",
    ["Tiền xử lý biển số xe", "Cải thiện ảnh vệ tinh GIS", "Nâng cao ảnh ánh sáng kém"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.subheader("Ảnh gốc")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

    # ------------------------------
    # 1. Tiền xử lý biển số xe
    # ------------------------------
    if option == "Tiền xử lý biển số xe":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # CLAHE để tăng tương phản
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)

        # Lọc nhiễu
        denoised = cv2.bilateralFilter(contrast, 11, 17, 17)

        # Canny edge
        edges = cv2.Canny(denoised, 30, 200)

        st.subheader("Ảnh sau tiền xử lý biển số xe")
        st.image(edges, caption="Edges để phát hiện biển số", use_container_width=True, channels="GRAY")

    # ------------------------------
    # 2. Cải thiện ảnh vệ tinh GIS
    # ------------------------------
    elif option == "Cải thiện ảnh vệ tinh GIS":
        # Cân bằng histogram
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        hist_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        # Unsharp Masking
        gaussian = cv2.GaussianBlur(hist_eq, (9, 9), 10)
        sharpened = cv2.addWeighted(hist_eq, 1.5, gaussian, -0.5, 0)

        # Bilateral filter
        smooth = cv2.bilateralFilter(sharpened, 9, 75, 75)

        st.subheader("Ảnh sau cải thiện GIS")
        st.image(cv2.cvtColor(smooth, cv2.COLOR_BGR2RGB), caption="Ảnh vệ tinh cải thiện", use_container_width=True)

    # ------------------------------
    # 3. Ảnh ánh sáng kém
    # ------------------------------
    elif option == "Nâng cao ảnh ánh sáng kém":
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(img_yuv)

        # CLAHE trên kênh Y (độ sáng)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        y_clahe = clahe.apply(y)

        img_yuv = cv2.merge((y_clahe, cr, cb))
        bright = cv2.cvtColor(img_yuv, cv2.COLOR_YCrCb2BGR)

        # Giảm nhiễu
        denoised = cv2.fastNlMeansDenoisingColored(bright, None, 10, 10, 7, 21)

        st.subheader("Ảnh sau cải thiện ánh sáng kém")
        st.image(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB), caption="Ảnh nâng cao ánh sáng", use_container_width=True)
