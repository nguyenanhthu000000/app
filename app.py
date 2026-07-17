import streamlit as st
import cv2
import numpy as np

def phep_mo_opening(binary_img, kernel):
    if binary_img is None:
        return None
    return cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

def phep_dong_closing(binary_img, kernel):
    if binary_img is None:
        return None
    return cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

def tao_nhieu_hat_trang(binary_img, phan_tram=0.03):
    noisy_img = binary_img.copy()
    num_noise = int(phan_tram * binary_img.size)
    coords = [np.random.randint(0, i - 1, num_noise) for i in binary_img.shape]
    noisy_img[tuple(coords)] = 255
    return noisy_img

def tao_vet_nut_den(binary_img):
    damaged_img = binary_img.copy()
    h, w = binary_img.shape
    cv2.line(damaged_img, (int(w*0.1), int(h*0.4)), (int(w*0.9), int(h*0.45)), 0, 6)
    cv2.circle(damaged_img, (int(w*0.5), int(h*0.65)), 18, 0, -1)
    cv2.circle(damaged_img, (int(w*0.35), int(h*0.25)), 12, 0, -1)
    return damaged_img

st.set_page_config(page_title="Đồ án Đề tài 19", layout="wide")

st.title("Ứng dụng Xử lý hình thái học Ảnh Nhị phân 🤖")
st.markdown("### **Đề tài 19: Ứng dụng phép mở và phép đóng trong xử lý ảnh nhị phân**")
st.markdown("---")

st.sidebar.header("⚙️ KHUNG ĐIỀU KHIỂN")
st.sidebar.subheader("Bước 1: Chọn Đề tài Khảo sát")
app_mode = st.sidebar.selectbox(
    "Lựa chọn phép toán toán hình thái học:",
    ["Giới thiệu đề tài", "Phép Mở (Opening) - Xóa nhiễu", "Phép Đóng (Closing) - Lắp vết nứt"]
)

st.sidebar.subheader("Bước 2: Cấu hình tham số Kernel")
kernel_size = st.sidebar.slider("Kích thước Mặt nạ (Kernel Size):", min_value=3, max_value=21, value=5, step=2)
kernel_type = st.sidebar.selectbox("Hình dáng Mặt nạ (Kernel Shape):", ["Hình chữ nhật (RECT)", "Hình chữ thập (CROSS)", "Hình Elip (ELLIPSE)"])

if kernel_type == "Hình chữ nhật (RECT)":
    k_shape = cv2.MORPH_RECT
elif kernel_type == "Hình chữ thập (CROSS)":
    k_shape = cv2.MORPH_CROSS
else:
    k_shape = cv2.MORPH_ELLIPSE

kernel = cv2.getStructuringElement(k_shape, (kernel_size, kernel_size))

if app_mode == "Giới thiệu đề tài":
    st.subheader("📋 Tổng quan Đề tài môn học")
    st.info("""
    **ĐỀ TÀI 19: ỨNG DỤNG PHÉP MỞ VÀ PHÉP ĐÓNG TRONG XỬ LÝ ẢNH NHỊ PHÂN**
    
    * **Mục tiêu:** Ứng dụng toán hình thái học để làm sạch ảnh, loại bỏ chi tiết thừa nhỏ li ti (nhiễu) và chữa lành các liên kết bị đứt nét, lỗ thủng nội bộ của vật thể.
    * **Hướng dẫn sử dụng:** Chọn menu tính năng 'Phép Mở' hoặc 'Phép Đóng' ở thanh điều hướng bên trái, sau đó tải ảnh lên để xem kết quả biến đổi thời gian thực.
    """)
    st.image("https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?auto=format&fit=crop&w=800&q=80", caption="Hệ thống phân tích xử lý ảnh thông minh số hóa")

else:
    st.subheader("📸 Tải ảnh đầu vào lên hệ thống")
    uploaded_file = st.file_uploader("Kéo và thả file ảnh vào đây (Hỗ trợ JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        src_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        _, base_binary = cv2.threshold(src_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image(base_binary, caption="1. Ảnh nhị phân gốc", use_container_width=True)
            
        if app_mode == "Phép Mở (Opening) - Xóa nhiễu":
            img_with_fault = tao_nhieu_hat_trang(base_binary)
            img_result = phep_mo_opening(img_with_fault, kernel)
            
            with col2:
                st.image(img_with_fault, caption="2. Ảnh giả lập dính nhiễu hạt trắng", use_container_width=True)
            with col3:
                st.image(img_result, caption=f"3. Kết quả Phép Mở (Kernel {kernel_size}x{kernel_size})", use_container_width=True)
                
            st.success("💡 **Phân tích của hệ thống:** Phép mở thực hiện phép co trước để bóc tách triệt tiêu các chấm nhiễu trắng nhỏ hơn kích thước mặt nạ, sau đó giãn ra để hồi phục cấu trúc vật thể ban đầu nguyên vẹn!")
            
        elif app_mode == "Phép Đóng (Closing) - Lắp vết nứt":
            img_with_fault = tao_vet_nut_den(base_binary)
            img_result = phep_dong_closing(img_with_fault, kernel)
            
            with col2:
                st.image(img_with_fault, caption="2. Ảnh giả lập bị nứt gãy / thủng lỗ đen", use_container_width=True)
            with col3:
                st.image(img_result, caption=f"3. Kết quả Phép Đóng (Kernel {kernel_size}x{kernel_size})", use_container_width=True)
                
            st.success("💡 **Phân tích của hệ thống:** Phép đóng thực hiện phép giãn trước để làm phình to vùng trắng nhằm lấp đầy hoàn toàn khe nứt hoặc hố đen tổn hại, sau đó co lại nhằm trả về đúng kích thước chuẩn xác ban đầu!")

    else:
        st.warning("⚠️ Đang đợi tải file ảnh lên để kích hoạt hệ thống xử lý tự động...")