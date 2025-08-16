import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.title("🖼️ Ứng dụng minh họa xử lý ảnh")

# Upload ảnh
uploaded_file = st.file_uploader("📂 Tải ảnh của bạn lên", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img = np.array(image)

    st.subheader("Ảnh gốc")
    st.image(img, caption="Ảnh gốc", use_column_width=True)

    # Chọn loại xử lý
    option = st.sidebar.radio(
        "📌 Chọn loại xử lý ảnh:",
        (
            "🚗 Tiền xử lý cho nhận dạng biển số xe",
            "🛰️ Cải thiện ảnh vệ tinh GIS",
            "🌙 Nâng cao chất lượng ảnh chụp ánh sáng kém",
        )
    )

    # ======================================
    # 1. Tiền xử lý ảnh biển số xe
    # ======================================
    if option == "🚗 Tiền xử lý cho nhận dạng biển số xe":
        st.subheader("🚗 Tiền xử lý cho nhận dạng biển số xe")

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        clip = st.slider("⚙️ Độ tương phản (CLAHE - Clip Limit)", 1.0, 5.0, 2.0)
        tile = st.slider("⚙️ Kích thước ô lưới (CLAHE - Grid Size)", 4, 16, 8)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
        clahe_img = clahe.apply(gray)

        d = st.slider("⚙️ Kích thước lọc nhiễu Bilateral (px)", 1, 15, 9)
        sigmaColor = st.slider("⚙️ Độ nhạy màu Bilateral", 10, 150, 75)
        sigmaSpace = st.slider("⚙️ Phạm vi không gian Bilateral", 10, 150, 75)
        bilateral = cv2.bilateralFilter(clahe_img, d, sigmaColor, sigmaSpace)

        low = st.slider("⚙️ Ngưỡng dưới Canny", 50, 200, 100)
        high = st.slider("⚙️ Ngưỡng trên Canny", 100, 300, 200)
        edges = cv2.Canny(bilateral, low, high)

        st.image([gray, clahe_img, bilateral, edges],
                 caption=["Grayscale", "CLAHE", "Bilateral", "Canny Edge"],
                 use_container_width=True, channels="GRAY")

    # ======================================
    # 2. Cải thiện ảnh vệ tinh
    # ======================================
    elif option == "🛰️ Cải thiện ảnh vệ tinh GIS":
        st.subheader("🛰️ Cải thiện ảnh vệ tinh GIS")

        ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y_eq = cv2.equalizeHist(y)
        ycrcb_eq = cv2.merge([y_eq, cr, cb])
        hist_eq = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2RGB)

        sigma = st.slider("⚙️ Độ mờ Gaussian (Unsharp - Sigma)", 1.0, 5.0, 1.5)
        amount = st.slider("⚙️ Mức tăng sắc nét (%)", 0.0, 3.0, 1.5)
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        unsharp = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)

        d = st.slider("⚙️ Kích thước lọc Bilateral (px)", 1, 15, 9)
        sigmaColor = st.slider("⚙️ Độ nhạy màu Bilateral", 10, 150, 75)
        sigmaSpace = st.slider("⚙️ Phạm vi không gian Bilateral", 10, 150, 75)
        bilateral = cv2.bilateralFilter(hist_eq, d, sigmaColor, sigmaSpace)

        st.image([hist_eq, unsharp, bilateral],
                 caption=["Cân bằng Histogram", "Tăng sắc nét (Unsharp)", "Làm mượt Bilateral"],
                 use_container_width=True)

    # ======================================
    # 3. Ảnh ánh sáng kém
    # ======================================
    elif option == "🌙 Nâng cao chất lượng ảnh chụp ánh sáng kém":
        st.subheader("🌙 Nâng cao ảnh chụp ánh sáng kém")

        ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        alpha = st.slider("⚙️ Hệ số tăng sáng (alpha)", 1.0, 3.0, 1.5)
        y_boost = cv2.convertScaleAbs(y, alpha=alpha)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y_clahe = clahe.apply(y_boost)
        ycrcb_eq = cv2.merge([y_clahe, cr, cb])
        img_clahe = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2RGB)

        h = st.slider("⚙️ Độ cao cửa sổ lọc (h)", 5, 20, 10)
        template = st.slider("⚙️ Kích thước cửa sổ so khớp", 3, 15, 7)
        search = st.slider("⚙️ Kích thước cửa sổ tìm kiếm", 15, 30, 21)
        denoised = cv2.fastNlMeansDenoisingColored(img_clahe, None, h, h, template, search)

        st.image([img, img_clahe, denoised],
                 caption=["Ảnh gốc", "Tăng sáng + CLAHE", "Giảm nhiễu"],
                 use_container_width=True)
