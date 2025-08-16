import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Demo xử lý ảnh với OpenCV 🎨")

# Upload ảnh
uploaded_file = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc ảnh bằng PIL rồi chuyển sang OpenCV
    image = Image.open(uploaded_file)
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # giữ chuẩn BGR cho xử lý

    # Hiển thị ảnh gốc (chuyển sang RGB khi show)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Ảnh gốc", use_container_width=True)

    st.subheader("Chọn loại xử lý ảnh")

    option = st.selectbox(
        "Chọn một bộ lọc:",
        ("Giữ nguyên", "Làm mờ Gaussian", "Làm sắc nét", "Canny Edge Detection")
    )

    # Xử lý theo lựa chọn
    if option == "Giữ nguyên":
        result = img
    elif option == "Làm mờ Gaussian":
        result = cv2.GaussianBlur(img, (15, 15), 0)
    elif option == "Làm sắc nét":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        result = cv2.filter2D(img, -1, kernel)
    elif option == "Canny Edge Detection":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = cv2.Canny(gray, 100, 200)
        # ảnh Canny là grayscale → show trực tiếp
        st.image(result, caption="Ảnh sau xử lý (Canny)", use_container_width=True)
        result = None  # tránh show thêm bên dưới

    # Hiển thị ảnh kết quả (nếu còn giữ màu)
    if result is not None:
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                 caption=f"Ảnh sau xử lý: {option}", use_container_width=True)
