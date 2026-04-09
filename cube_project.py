import cv2
import numpy as np
import os
import sys

# --- 1. 配置路径与参数 ---
PARAM_PATH = '/home/gsp/00fish_eye_camera/fisheye_calib_params.npz'
SAVE_DIR = '/home/gsp/00fish_eye_camera/cubemap_photos'
CAMERA_ID = 2

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# --- 2. 加载参数 ---
try:
    with np.load(PARAM_PATH) as data:
        K = data['K']
        D = data['D']
    print("✅ 成功加载相机内参和畸变系数！")
except FileNotFoundError:
    print(f"❌ 找不到参数文件：{PARAM_PATH}")
    sys.exit()

# --- 3. 核心算法：构建立方体 6 个面的映射表 ---
def build_cubemap_maps(K, D, face_size):
    print(f"正在构建立方体投影映射表 (每个面 {face_size}x{face_size})...")
    
    # a. 创建一个完美的 90 度 FOV 的针孔相机内参
    # 视场角为 90 度时，焦距 f 正好等于图像边长的一半
    f = face_size / 2.0
    c = face_size / 2.0
    K_new = np.array([
        [f, 0, c],
        [0, f, c],
        [0, 0, 1]
    ], dtype=np.float32)

    # b. 定义 6 个面相对于原始相机的 3D 旋转矩阵 (核心数学)
    # R_inv 的每一列代表新相机的 [X轴, Y轴, Z轴] 在原相机坐标系下的方向
    # 假设原相机：Z向前，Y向下，X向右
    R_inv_front  = np.array([[ 1,  0,  0], [ 0,  1,  0], [ 0,  0,  1]]) # 前
    R_inv_right  = np.array([[ 0,  0,  1], [ 0,  1,  0], [-1,  0,  0]]) # 右
    R_inv_left   = np.array([[ 0,  0, -1], [ 0,  1,  0], [ 1,  0,  0]]) # 左
    R_inv_top    = np.array([[ 1,  0,  0], [ 0,  0, -1], [ 0,  1,  0]]) # 上
    R_inv_bottom = np.array([[ 1,  0,  0], [ 0,  0,  1], [ 0, -1,  0]]) # 下
    R_inv_back   = np.array([[-1,  0,  0], [ 0,  1,  0], [ 0,  0, -1]]) # 后

    # 取逆矩阵得到供 OpenCV 使用的 R
    rotations = {
        'front':  np.linalg.inv(R_inv_front),
        'right':  np.linalg.inv(R_inv_right),
        'left':   np.linalg.inv(R_inv_left),
        'top':    np.linalg.inv(R_inv_top),
        'bottom': np.linalg.inv(R_inv_bottom),
        'back':   np.linalg.inv(R_inv_back)
    }

    maps = {}
    # c. 为每个面生成映射表
    for name, R in rotations.items():
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, R, K_new, (face_size, face_size), cv2.CV_32FC1
        )
        maps[name] = (map1, map2)
    
    return maps

# --- 4. 初始化 ---
FACE_SIZE = 400 # 每个正方形面的分辨率，你可以调大(如 600)获取更高清画质
maps = build_cubemap_maps(K, D, FACE_SIZE)

cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print("❌ 无法打开摄像头")
    sys.exit()

print("\n--- 📷 立方体展开 (Cubemap) 已启动 ---")
img_count = 0

# --- 5. 实时渲染循环 ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. 极速提取 6 个面
    faces = {}
    for name in ['front', 'right', 'left', 'top', 'bottom', 'back']:
        faces[name] = cv2.remap(frame, maps[name][0], maps[name][1], 
                                interpolation=cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_CONSTANT)

        # 给每个面打上文字标签，方便你认出来
        cv2.putText(faces[name], name.upper(), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 2. 拼装成经典的十字架 (Cross) 形状
    # 画布大小：宽 4 个面，高 3 个面
    cross_img = np.zeros((FACE_SIZE * 3, FACE_SIZE * 4, 3), dtype=np.uint8)

    # 按照十字架布局填入图像
    cross_img[0:FACE_SIZE, FACE_SIZE:FACE_SIZE*2] = faces['top']
    cross_img[FACE_SIZE:FACE_SIZE*2, 0:FACE_SIZE] = faces['left']
    cross_img[FACE_SIZE:FACE_SIZE*2, FACE_SIZE:FACE_SIZE*2] = faces['front']
    cross_img[FACE_SIZE:FACE_SIZE*2, FACE_SIZE*2:FACE_SIZE*3] = faces['right']
    cross_img[FACE_SIZE:FACE_SIZE*2, FACE_SIZE*3:FACE_SIZE*4] = faces['back']
    cross_img[FACE_SIZE*2:FACE_SIZE*3, FACE_SIZE:FACE_SIZE*2] = faces['bottom']

    # 3. 显示缩小版以防爆屏
    DISPLAY_SCALE = 0.6
    preview_w = int(cross_img.shape[1] * DISPLAY_SCALE)
    preview_h = int(cross_img.shape[0] * DISPLAY_SCALE)
    preview_frame = cv2.resize(cross_img, (preview_w, preview_h))

    cv2.imshow('Cubemap Projection (Cross Layout)', preview_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        save_name = os.path.join(SAVE_DIR, f"cubemap_{img_count:04d}.jpg")
        cv2.imwrite(save_name, cross_img) # 保存高清大图
        print(f"📸 保存成功: {save_name}")
        img_count += 1
    elif key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()