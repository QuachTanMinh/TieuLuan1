import cv2
import numpy as np
import streamlit as st

# ===== Các hàm xử lý =====
def negative(img): return 255 - img

def log_transform(img):
    c = 255 / np.log(1 + np.max(img))
    return np.uint8(c * np.log(img.astype(np.float64) + 1))

def gamma_correction(img, gamma=0.5):
    return np.array(255 * (img/255.0)**gamma, dtype='uint8')

def hist_equalization(img):
    return cv2.equalizeHist(img)

def clahe_equalization(img, clip=2.0, tile=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    return clahe.apply(img)

# ===== Giao diện Streamlit =====
st.title("Demo xử lý ảnh - Point Processing")

uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    st.image(img, caption="Ảnh gốc", use_container_width=True)

    option = st.selectbox("Chọn phương pháp xử lý", 
        ["Negative", "Log Transform", "Gamma Correction", "Histogram Equalization", "CLAHE"])

    if option == "Negative":
        result = negative(img)
    elif option == "Log Transform":
        result = log_transform(img)
    elif option == "Gamma Correction":
        gamma = st.slider("Chọn gamma", 0.1, 3.0, 0.5)
        result = gamma_correction(img, gamma)
    elif option == "Histogram Equalization":
        result = hist_equalization(img)
    elif option == "CLAHE":
        clip = st.slider("clipLimit", 1.0, 5.0, 2.0)
        result = clahe_equalization(img, clip)

    st.image(result, caption="Ảnh sau xử lý", use_container_width=True)
