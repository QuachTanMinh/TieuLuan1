import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Xử lý ảnh nâng cao", layout="wide")
st.title("📷 Demo Xử Lý Ảnh với OpenCV + Streamlit")

# ---------------------------
# Utils
# ---------------------------
def load_image_rgb(uploaded_file) -> np.ndarray:
    """
    Đảm bảo ảnh luôn là RGB 3 kênh (tránh RGBA/LA/P mode gây lỗi cv2.cvtColor).
    """
    img = Image.open(uploaded_file).convert("RGB")
    return np.array(img)  # RGB

def show_pair(img_left, img_right, cap_left="Ảnh gốc", cap_right="Ảnh sau xử lý"):
    c1, c2 = st.columns(2)
    with c1:
        st.image(img_left, caption=cap_left, use_container_width=True)
    with c2:
        st.image(img_right, caption=cap_right, use_container_width=True)

def unsharp_mask_rgb(img_rgb: np.ndarray, radius: float = 1.5, amount: float = 1.0) -> np.ndarray:
    """
    Unsharp Mask cho ảnh màu RGB.
    radius: sigma cho Gaussian blur (dùng (0,0) để tự suy ra kernel từ sigma).
    amount: hệ số làm nét, 1.0 ~ nhẹ, 1.5–2.0 ~ mạnh.
    """
    blur = cv2.GaussianBlur(img_rgb, (0, 0), radius)
    sharp = cv2.addWeighted(img_rgb, 1.0 + amount, blur, -amount, 0)
    return sharp

# ---------------------------
# Giao diện
# ---------------------------
task = st.sidebar.selectbox(
    "Chọn bài toán:",
    ("🚗 Tiền xử lý nhận dạng biển số xe",
     "🛰️ Cải thiện ảnh vệ tinh GIS",
     "🌙 Nâng cao chất lượng ảnh chụp ánh sáng kém")
)

uploaded_file = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("⬆️ Hãy tải lên một ảnh để bắt đầu.")
    st.stop()

# Luôn ép về RGB 3 kênh để tránh lỗi cv2.cvtColor
img_rgb = load_image_rgb(uploaded_file)
st.image(img_rgb, caption="Ảnh gốc (RGB)", use_container_width=True)

# ======================================================
# 🚗 1) TIỀN XỬ LÝ NHẬN DẠNG BIỂN SỐ XE
# ======================================================
if task == "🚗 Tiền xử lý nhận dạng biển số xe":
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # CLAHE trên ảnh xám
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    # Bilateral filter giữ biên
    gray_bi = cv2.bilateralFilter(gray_clahe, d=9, sigmaColor=75, sigmaSpace=75)

    # Canny
    edges = cv2.Canny(gray_bi, 100, 200)

    st.subheader("Pipeline biển số xe")
    st.image([gray, gray_clahe, gray_bi, edges],
             caption=["Grayscale", "CLAHE (xám)", "Bilateral", "Canny"],
             use_container_width=True)

# ======================================================
# 🛰️ 2) CẢI THIỆN ẢNH VỆ TINH GIS (ĐÃ FIX)
# ======================================================
elif task == "🛰️ Cải thiện ảnh vệ tinh GIS":
    # 1) Cân bằng histogram trên kênh Y (giữ màu đúng)
    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    Y_eq = cv2.equalizeHist(Y)  # yêu cầu ảnh 8-bit 1 kênh -> OK
    img_eq = cv2.cvtColor(cv2.merge((Y_eq, Cr, Cb)), cv2.COLOR_YCrCb2RGB)

    # 2) Unsharp Masking để tăng độ sắc nét (trên ảnh màu)
    radius = st.sidebar.slider("Unsharp radius (sigma)", 0.5, 3.0, 1.2, 0.1)
    amount = st.sidebar.slider("Unsharp amount", 0.2, 2.5, 0.8, 0.1)
    img_sharp = unsharp_mask_rgb(img_eq, radius=radius, amount=amount)

    # 3) Bilateral filter để làm mượt nhưng giữ biên
    d = st.sidebar.slider("Bilateral d (pixels)", 5, 15, 9, 1)
    sColor = st.sidebar.slider("Bilateral sigmaColor", 10, 150, 50, 5)
    sSpace = st.sidebar.slider("Bilateral sigmaSpace", 10, 150, 50, 5)
    img_final = cv2.bilateralFilter(img_sharp, d=d, sigmaColor=sColor, sigmaSpace=sSpace)

    st.subheader("Pipeline ảnh vệ tinh (màu)")
    st.image([img_eq, img_sharp, img_final],
             caption=["Equalize kênh Y (giữ màu)", "Unsharp Masking", "Bilateral (kết quả)"],
             use_container_width=True)

# ======================================================
# 🌙 3) ẢNH ÁNH SÁNG KÉM (GIỮ MÀU)
# ======================================================
elif task == "🌙 Nâng cao chất lượng ảnh chụp ánh sáng kém":
    # Làm sáng & tăng tương phản trên kênh Y để không lệch màu
    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)

    # Tăng sáng nhẹ trước khi CLAHE (alpha>1 -> sáng hơn)
    alpha = st.sidebar.slider("Tăng sáng (alpha)", 1.0, 2.5, 1.3, 0.1)
    Y_bright = cv2.convertScaleAbs(Y, alpha=alpha, beta=0)

    clahe_clip = st.sidebar.slider("CLAHE clipLimit", 1.0, 5.0, 3.0, 0.1)
    tile = st.sidebar.slider("CLAHE tileGrid", 4, 16, 8, 1)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(tile, tile))
    Y_clahe = clahe.apply(Y_bright)

    img_clahe = cv2.cvtColor(cv2.merge((Y_clahe, Cr, Cb)), cv2.COLOR_YCrCb2RGB)

    # Khử nhiễu màu (NLM) trên ảnh màu
    h = st.sidebar.slider("Denoise h (luminance)", 5, 20, 10, 1)
    hc = st.sidebar.slider("Denoise hColor (chrominance)", 5, 20, 10, 1)
    denoised = cv2.fastNlMeansDenoisingColored(img_clahe, None, h, hc, 7, 21)

    show_pair(img_rgb, denoised, "Ảnh gốc (RGB)", "Sau cải thiện ánh sáng kém")
