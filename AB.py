import cv2
import numpy as np
import os
import sys
from ultralytics import YOLO

# --- 1. 配置路径与参数 ---
PARAM_PATH = '/home/gsp/00fish_eye_camera/fisheye_calib_params.npz'
SAVE_DIR = '/home/gsp/00fish_eye_camera/yolo_comparison'
CAMERA_ID = 0

# 你的 YOLO-Pose 模型路径
YOLO_MODEL_PATH = '/home/gsp/00fish_eye_camera/best.pt' 

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# --- 2. 加载参数 ---
try:
    with np.load(PARAM_PATH) as data:
        K = data['K']
        D = data['D']
except FileNotFoundError:
    print(f"❌ 找不到参数文件：{PARAM_PATH}")
    sys.exit()

# --- 3. 加载模型 ---
print(f"正在加载 YOLO 模型: {YOLO_MODEL_PATH} ...")
try:
    model = YOLO(YOLO_MODEL_PATH)
except Exception as e:
    print(f"❌ YOLO 模型加载失败: {e}")
    sys.exit()

# --- 4. 映射表计算函数 ---
def build_equirectangular_map(K, D, out_w, out_h, fov_x_deg, fov_y_deg):
    fov_x = np.deg2rad(fov_x_deg)
    fov_y = np.deg2rad(fov_y_deg)
    x, y = np.meshgrid(np.arange(out_w), np.arange(out_h))
    yaw = (x / out_w - 0.5) * fov_x
    pitch = (y / out_h - 0.5) * fov_y
    X = np.sin(yaw) * np.cos(pitch)
    Y = np.sin(pitch)
    Z = np.cos(yaw) * np.cos(pitch)
    points_3d = np.stack((X, Y, Z), axis=-1).reshape(-1, 1, 3).astype(np.float64)
    points_2d, _ = cv2.fisheye.projectPoints(points_3d, np.zeros((3,1)), np.zeros((3,1)), K, D)
    points_2d = points_2d.reshape(out_h, out_w, 2)
    return points_2d[..., 0].astype(np.float32), points_2d[..., 1].astype(np.float32)

# --- 5. 初始化 ---
cap = cv2.VideoCapture(CAMERA_ID)
ret, frame = cap.read()
if not ret:
    sys.exit()
h, w = frame.shape[:2]

OUT_W = 1280
OUT_H = 900
FOV_X = 180.0 
FOV_Y = 120.0 

map1, map2 = build_equirectangular_map(K, D, OUT_W, OUT_H, FOV_X, FOV_Y)

print("\n--- 📷 YOLO 终极对比模式 已启动 ---")
print("👉 按 's' 键 : 保存对比画面")
print("👉 按 'q' 键 : 退出\n")
img_count = 0

# --- 6. 实时循环 ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. 生成全景图
    panorama_frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # ==========================================
    # 2. 【核心修改】YOLO 双重推理对比
    # ==========================================
    # 🥊 第一回合：让 YOLO 直接挑战扭曲的鱼眼原图
    results_raw = model(frame, conf=0.5, verbose=False)
    annotated_raw = results_raw[0].plot()

    # 🥊 第二回合：让 YOLO 识别处理好的等距柱状全景图
    results_pano = model(panorama_frame, conf=0.5, verbose=False)
    annotated_pano = results_pano[0].plot()

    # 3. 缩放左侧原图以便完美拼接
    orig_aspect_ratio = w / h
    new_orig_w = int(OUT_H * orig_aspect_ratio)
    resized_annotated_raw = cv2.resize(annotated_raw, (new_orig_w, OUT_H))

    # 4. 左右拼接 (左：原图直接识别，右：全景图识别)
    combined_frame = np.hstack((resized_annotated_raw, annotated_pano))

    # 5. 打上标签
    cv2.putText(combined_frame, "YOLO on Raw Fisheye (Distorted)", (10, OUT_H - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(combined_frame, "YOLO on Equirectangular (Flattened)", (new_orig_w + 10, OUT_H - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 6. 缩放预览窗口
    DISPLAY_SCALE = 0.7  
    preview_w = int(combined_frame.shape[1] * DISPLAY_SCALE)
    preview_h = int(combined_frame.shape[0] * DISPLAY_SCALE)
    preview_frame = cv2.resize(combined_frame, (preview_w, preview_h))

    cv2.imshow('YOLO Performance A/B Test', preview_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        save_name = os.path.join(SAVE_DIR, f"ab_test_{img_count:04d}.jpg")
        cv2.imwrite(save_name, combined_frame)  
        print(f"📸 战况保存成功: {save_name}")
        img_count += 1
    elif key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()