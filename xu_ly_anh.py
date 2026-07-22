"""
xu_ly_anh.py
Xử lý Hình thái học Ảnh Nhị Phân
Chứa các hàm giải quyết trọng tâm bài toán (Đề tài 19):
    - Phép Mở (Opening)  : loại bỏ nhiễu hạt nhỏ
    - Phép Đóng (Closing): lấp lỗ, nối vết đứt
    - Các hàm tạo dữ liệu mô phỏng (nhiễu, vết nứt) để demo
"""

import cv2
import numpy as np


def phep_mo_opening(binary_img, kernel):
    """
    Phép Mở (Opening) = Erosion sau đó Dilation.
    Dùng để loại bỏ các đốm nhiễu nhỏ (hạt trắng) mà vẫn giữ
    được hình dạng chính của vật thể lớn trong ảnh.
    """
    if binary_img is None:
        return None
    return cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)


def phep_dong_closing(binary_img, kernel):
    """
    Phép Đóng (Closing) = Dilation sau đó Erosion.
    Dùng để lấp đầy các lỗ nhỏ, khe nứt hoặc vết đứt bên trong
    vật thể, giúp khôi phục sự liền mạch.
    """
    if binary_img is None:
        return None
    return cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)


def tao_nhieu_hat_trang(binary_img, phan_tram=0.03, seed=42):
    """
    Mô phỏng nhiễu muối-tiêu (salt noise): rải ngẫu nhiên các
    điểm ảnh trắng lên ảnh nhị phân gốc
    """
    rng = np.random.default_rng(seed)
    noisy_img = binary_img.copy()
    num_noise = int(phan_tram * binary_img.size)
    coords = [rng.integers(0, i - 1, num_noise) for i in binary_img.shape]
    noisy_img[tuple(coords)] = 255
    return noisy_img


def tao_vet_nut_den(binary_img):
    """
    Mô phỏng vật thể bị hỏng: vẽ thêm 1 đường nứt (line) và
    2 lỗ thủng (circle) màu đen lên ảnh\
    """
    damaged_img = binary_img.copy()
    h, w = binary_img.shape
    cv2.line(damaged_img, (int(w * 0.1), int(h * 0.4)), (int(w * 0.9), int(h * 0.45)), 0, 6)
    cv2.circle(damaged_img, (int(w * 0.5), int(h * 0.65)), 18, 0, -1)
    cv2.circle(damaged_img, (int(w * 0.35), int(h * 0.25)), 12, 0, -1)
    return damaged_img