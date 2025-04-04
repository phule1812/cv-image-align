import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

# --- 1. Đọc ảnh và chuyển sang RGB (nếu cần) ---
ref_filename = "documents/img.png"
im1 = imread(ref_filename)
im2_filename = "documents/img2.png"
im2 = imread(im2_filename)

# Nếu ảnh ở dạng uint8 thì chuyển về float [0,1]
if im1.dtype == np.uint8:
    im1 = im1.astype(np.float32) / 255.0
if im2.dtype == np.uint8:
    im2 = im2.astype(np.float32) / 255.0

# Hiển thị ảnh ban đầu
plt.figure(figsize=(15, 8))
plt.subplot(121)
plt.imshow(im1)
plt.axis('off')
plt.title('Original Image')
plt.subplot(122)
plt.imshow(im2)
plt.axis('off')
plt.title('Image to be aligned')
plt.show()

# --- 2. Chuyển đổi ảnh sang grayscale ---
def rgb2gray(im):
    return 0.2989 * im[:, :, 0] + 0.5870 * im[:, :, 1] + 0.1140 * im[:, :, 2]

im1_gray = rgb2gray(im1)
im2_gray = rgb2gray(im2)

# --- 3. Phát hiện điểm góc bằng Harris ---
def harris_corners(img, window_size=3, k=0.04):
    # Tính đạo hàm theo x, y
    Ix = np.gradient(img, axis=1)
    Iy = np.gradient(img, axis=0)
    
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    
    height, width = img.shape
    offset = window_size // 2
    R = np.zeros_like(img)
    
    # Tính corner response cho mỗi pixel (không xử lý biên)
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            Sxx = Ixx[y - offset:y + offset + 1, x - offset:x + offset + 1].sum()
            Syy = Iyy[y - offset:y + offset + 1, x - offset:x + offset + 1].sum()
            Sxy = Ixy[y - offset:y + offset + 1, x - offset:x + offset + 1].sum()
            det = Sxx * Syy - Sxy * Sxy
            trace = Sxx + Syy
            R[y, x] = det - k * (trace ** 2)
    return R

R1 = harris_corners(im1_gray)
R2 = harris_corners(im2_gray)

# Hàm lấy keypoints từ ma trận phản ứng
def get_keypoints(R, threshold_ratio=0.01, max_points=500):
    threshold = R.max() * threshold_ratio
    coords = np.argwhere(R > threshold)
    responses = R[coords[:, 0], coords[:, 1]]
    # Sắp xếp theo giá trị phản ứng giảm dần
    indices = np.argsort(-responses)
    coords = coords[indices]
    return coords[:max_points]

keypoints1 = get_keypoints(R1)
keypoints2 = get_keypoints(R2)

# --- 4. Trích xuất descriptor đơn giản (patch 9x9 quanh keypoint) ---
def extract_descriptor(img, keypoints, patch_size=9):
    descriptors = []
    half_size = patch_size // 2
    # Pad ảnh để tránh lỗi biên
    padded = np.pad(img, ((half_size, half_size), (half_size, half_size)), mode='constant')
    for y, x in keypoints:
        patch = padded[y:y + patch_size, x:x + patch_size]
        descriptors.append(patch.flatten())
    return np.array(descriptors)

descriptors1 = extract_descriptor(im1_gray, keypoints1)
descriptors2 = extract_descriptor(im2_gray, keypoints2)

# --- 5. Ghép nối các descriptor dùng SSD ---
def match_descriptors(desc1, desc2):
    matches = []
    for i, d1 in enumerate(desc1):
        # Tính SSD giữa d1 và tất cả descriptor của ảnh 2
        ssd = np.sum((desc2 - d1) ** 2, axis=1)
        j = np.argmin(ssd)
        matches.append((i, j, ssd[j]))
    # Sắp xếp theo khoảng cách (SSD) tăng dần
    matches = sorted(matches, key=lambda x: x[2])
    return matches

matches = match_descriptors(descriptors1, descriptors2)
num_good = int(len(matches) * 0.1)
good_matches = matches[:num_good]

# --- Hiển thị keypoints lên ảnh ---
def plot_keypoints(img, keypoints, title):
    plt.imshow(img, cmap='gray')
    plt.scatter(keypoints[:, 1], keypoints[:, 0], marker='o', facecolors='none', edgecolors='r')
    plt.axis('off')
    plt.title(title)

plt.figure(figsize=(15, 8))
plt.subplot(121)
plot_keypoints(im1_gray, keypoints1, 'Original Image Keypoints')
plt.subplot(122)
plot_keypoints(im2_gray, keypoints2, 'Image to be aligned Keypoints')
plt.show()

# --- Hiển thị các cặp ghép nối ---
def draw_matches(img1, img2, kp1, kp2, matches):
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    height = max(h1, h2)
    new_img = np.zeros((height, w1 + w2))
    new_img[:h1, :w1] = img1
    new_img[:h2, w1:] = img2
    
    plt.figure(figsize=(15, 8))
    plt.imshow(new_img, cmap='gray')
    for m in matches:
        i, j, _ = m
        y1, x1 = kp1[i]
        y2, x2 = kp2[j]
        plt.plot([x1, x2 + w1], [y1, y2], 'r-', linewidth=1)
    plt.axis('off')
    plt.title('Matches')
    plt.show()

draw_matches(im1_gray, im2_gray, keypoints1, keypoints2, good_matches)

# --- 6. Ước tính ma trận Homography bằng DLT và RANSAC ---
# Chuẩn bị tập điểm tương ứng. Lưu ý: chuyển (row, col) -> (x,y)
pts1 = np.array([keypoints1[i] for i, j, _ in good_matches])[:, ::-1]  # (x,y) từ ảnh tham chiếu
pts2 = np.array([keypoints2[j] for i, j, _ in good_matches])[:, ::-1]  # (x,y) từ ảnh cần căn chỉnh

def compute_homography(pts_src, pts_dst):
    # pts_src, pts_dst: mảng Nx2
    N = pts_src.shape[0]
    A = []
    for i in range(N):
        x, y = pts_src[i, 0], pts_src[i, 1]
        u, v = pts_dst[i, 0], pts_dst[i, 1]
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    H = h.reshape(3, 3)
    return H / H[2, 2]

def ransac_homography(pts_src, pts_dst, iterations=1000, threshold=5):
    best_H = None
    max_inliers = 0
    best_inliers = None
    N = pts_src.shape[0]
    for _ in range(iterations):
        idx = np.random.choice(N, 4, replace=False)
        H = compute_homography(pts_src[idx], pts_dst[idx])
        pts_src_h = np.concatenate([pts_src, np.ones((N, 1))], axis=1)
        pts_trans = (H @ pts_src_h.T).T
        pts_trans = pts_trans[:, :2] / pts_trans[:, 2:3]
        distances = np.linalg.norm(pts_dst - pts_trans, axis=1)
        inliers = distances < threshold
        count = np.sum(inliers)
        if count > max_inliers:
            max_inliers = count
            best_H = H
            best_inliers = inliers
    if best_inliers is not None and np.sum(best_inliers) >= 4:
        best_H = compute_homography(pts_src[best_inliers], pts_dst[best_inliers])
    return best_H, best_inliers

# Ở đây ta muốn ánh xạ các điểm từ ảnh cần căn chỉnh (im2) về ảnh tham chiếu (im1)
H, inliers = ransac_homography(pts2, pts1, iterations=1000, threshold=5)

# --- 7. Biến đổi ảnh theo ma trận Homography ---
def warp_perspective(img, H, output_shape):
    h_out, w_out = output_shape
    # Tạo lưới tọa độ của ảnh đầu ra
    xx, yy = np.meshgrid(np.arange(w_out), np.arange(h_out))
    grid = np.stack([xx.ravel(), yy.ravel(), np.ones_like(xx.ravel())], axis=0)
    # Áp dụng ánh xạ ngược
    H_inv = np.linalg.inv(H)
    src_coords = H_inv @ grid
    src_coords = src_coords / src_coords[2, :]
    src_x = src_coords[0, :].reshape(h_out, w_out)
    src_y = src_coords[1, :].reshape(h_out, w_out)
    
    # Nội suy nearest neighbor
    src_x_nn = np.round(src_x).astype(int)
    src_y_nn = np.round(src_y).astype(int)
    
    h_src, w_src = img.shape[0], img.shape[1]
    warped = np.zeros((h_out, w_out, img.shape[2]), dtype=img.dtype)
    for i in range(h_out):
        for j in range(w_out):
            x = src_x_nn[i, j]
            y = src_y_nn[i, j]
            if 0 <= x < w_src and 0 <= y < h_src:
                warped[i, j] = img[y, x]
    return warped

h, w, _ = im1.shape
im2_reg = warp_perspective(im2, H, (h, w))

# --- 8. Hiển thị kết quả căn chỉnh ---
plt.figure(figsize=(15, 8))
plt.subplot(121)
plt.imshow(im1)
plt.axis('off')
plt.title('Original Image')
plt.subplot(122)
plt.imshow(im2_reg)
plt.axis('off')
plt.title('Aligned Image')
plt.show()
