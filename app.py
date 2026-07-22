import base64
import time
import uuid
import cv2
import numpy as np
from flask import Flask, render_template, request, session

from xu_ly_anh import (
    phep_mo_opening,
    phep_dong_closing,
    tao_nhieu_hat_trang,
    tao_vet_nut_den,
)

app = Flask(__name__)
app.secret_key = "morphlab_secret_key_pro"

# Bộ nhớ tạm phía server để lưu ảnh gốc theo từng phiên
# Cookie chỉ lưu 1 chuỗi ID nhỏ -> tránh lỗi vượt giới hạn 4KB của cookie
# Cấu trúc: {sid: (file_bytes, last_used_timestamp)}
IMAGE_CACHE = {}

CACHE_TTL_GIAY = 30 * 60


def don_dep_cache_cu():
    now = time.time()
    het_han = [sid for sid, (_, t) in IMAGE_CACHE.items() if now - t > CACHE_TTL_GIAY]
    for sid in het_han:
        IMAGE_CACHE.pop(sid, None)


def anh_sang_base64(img):
    if img is None:
        return None
    ok, buffer = cv2.imencode(".png", img)
    if not ok:
        return None
    return "data:image/png;base64," + base64.b64encode(buffer).decode("utf-8")


@app.route("/", methods=["GET", "POST"])
def index():
    don_dep_cache_cu()

    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    sid = session["sid"]


    if request.method == "GET":
        IMAGE_CACHE.pop(sid, None)

    # ---- Giá trị mặc định / lấy từ form ----
    app_mode = request.form.get("app_mode", "Giới thiệu đề tài")
    kernel_size = int(request.form.get("kernel_size", 5))
    kernel_type = request.form.get("kernel_type", "Hình chữ nhật (RECT)")
    invert_binary = request.form.get("invert_binary") == "on"

    if request.method == "POST" and request.form.get("clear_cache") == "1":
        IMAGE_CACHE.pop(sid, None)
        return render_template(
            "index.html",
            app_mode=app_mode,
            kernel_size=kernel_size,
            kernel_type=kernel_type,
            invert_binary=invert_binary,
            ket_qua=None,
            loi=None,
            da_co_anh=False,
        )

    if kernel_type == "Hình chữ nhật (RECT)":
        k_shape = cv2.MORPH_RECT
    elif kernel_type == "Hình chữ thập (CROSS)":
        k_shape = cv2.MORPH_CROSS
    else:
        k_shape = cv2.MORPH_ELLIPSE

    kernel = cv2.getStructuringElement(k_shape, (kernel_size, kernel_size))

    ket_qua = None
    loi = None

    if request.method == "POST" and app_mode != "Giới thiệu đề tài":
        uploaded_file = request.files.get("anh_dau_vao")

        # 1. Nếu người dùng chọn file mới -> Đọc và lưu vào server
        if uploaded_file and uploaded_file.filename != "":
            file_bytes = uploaded_file.read()
            src_img_check = cv2.imdecode(np.frombuffer(file_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if src_img_check is not None:
                IMAGE_CACHE[sid] = (file_bytes, time.time())
            else:
                loi = "Không đọc được file ảnh. Vui lòng thử lại với file khác."

        # 2. Nếu không tải file mới, dùng lại ảnh đã lưu trước đó theo sid
        src_img = None
        if sid in IMAGE_CACHE:
            img_bytes, _ = IMAGE_CACHE[sid]
            IMAGE_CACHE[sid] = (img_bytes, time.time())  # cập nhật thời điểm dùng gần nhất
            src_img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        # 3. Xử lý logic
        dang_chi_doi_mode = request.form.get("mode_switch") == "1"
        if src_img is None and not loi:
            # Nếu người dùng chỉ vừa đổi mode (chưa từng chọn ảnh) thì không
            # báo lỗi ngay, để trang chính chuyển sang khung "chưa có ảnh"
            # một cách bình thường thay vì hiện lỗi đỏ
            if not dang_chi_doi_mode:
                loi = "Vui lòng chọn một file ảnh trước khi xử lý."
        elif src_img is not None:
            thresh_mode = cv2.THRESH_BINARY_INV if invert_binary else cv2.THRESH_BINARY
            _, base_binary = cv2.threshold(src_img, 0, 255, thresh_mode + cv2.THRESH_OTSU)

            if app_mode == "Phép Mở (Opening) - Xóa nhiễu":
                img_with_fault = tao_nhieu_hat_trang(base_binary)
                img_result = phep_mo_opening(img_with_fault, kernel)
                nhan_anh_giua = "Ảnh có nhiễu hạt trắng"
                nhan_ket_qua = f"Kết quả Phép Mở ({kernel_size}×{kernel_size})"
                phan_tich = (
                    "Phép Mở giúp loại bỏ hoàn toàn nhiễu hạt nhỏ trong khi vẫn "
                    "giữ được hình dạng chính của vật thể."
                )
            else:  # Phép Đóng
                img_with_fault = tao_vet_nut_den(base_binary)
                img_result = phep_dong_closing(img_with_fault, kernel)
                nhan_anh_giua = "Ảnh có vết nứt / lỗ thủng"
                nhan_ket_qua = f"Kết quả Phép Đóng ({kernel_size}×{kernel_size})"
                phan_tich = (
                    "Phép Đóng giúp lấp đầy các khe nứt và lỗ thủng, khôi phục "
                    "sự liền mạch của vật thể."
                )

            ket_qua = {
                "anh_goc": anh_sang_base64(base_binary),
                "anh_giua": anh_sang_base64(img_with_fault),
                "anh_ket_qua": anh_sang_base64(img_result),
                "nhan_anh_giua": nhan_anh_giua,
                "nhan_ket_qua": nhan_ket_qua,
                "phan_tich": phan_tich,
                "kernel_size": kernel_size,
                "kernel_type": kernel_type.split(" (")[0],
                "kich_thuoc_goc": f"{src_img.shape[1]} × {src_img.shape[0]} px",
            }


    da_co_anh = sid in IMAGE_CACHE

    return render_template(
        "index.html",
        app_mode=app_mode,
        kernel_size=kernel_size,
        kernel_type=kernel_type,
        invert_binary=invert_binary,
        ket_qua=ket_qua,
        loi=loi,
        da_co_anh=da_co_anh,
    )

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    # Tắt use_reloader để Flask không tự nhân đôi process
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
