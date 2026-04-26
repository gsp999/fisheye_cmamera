import cv2
import numpy as np
import os
import sys
from ultralytics import YOLO  # 【新增】导入 YOLO 库

# --- 1. 配置路径与参数 ---
PARAM_PATH = '/home/gsp/00fish_eye_camera/fisheye_calib_params.npz'
SAVE_DIR = '/home/gsp/00fish_eye_camera/yolo_pose_results'
CAMERA_ID = 2

# 【新增】你的 YOLO-Pose 模型路径 (请替换为你自己的 .pt 文件路径)
YOLO_MODEL_PATH = '/home/gsp/00fish_eye_camera/best.pt' 

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# --- 2. 加载相机“黄金参数” ---
try:
    with np.load(PARAM_PATH) as data:
        K = data['K']
        D = data['D']
    print("✅ 成功加载相机内参和畸变系数！")
except FileNotFoundError:
    print(f"❌ 找不到参数文件：{PARAM_PATH}")
    sys.exit()

# --- 3. 加载 YOLO-Pose 模型 ---
print(f"正在加载 YOLO 模型: {YOLO_MODEL_PATH} ...")
try:
    model = YOLO(YOLO_MODEL_PATH)
    print("✅ YOLO 模型加载成功！")
except Exception as e:
    print(f"❌ YOLO 模型加载失败，请检查路径是否正确。\n错误信息: {e}")
    sys.exit()

# --- 4. 定义等距柱状投影的映射计算函数 ---
def build_equirectangular_map(K, D, out_w, out_h, fov_x_deg, fov_y_deg):
    print(f"正在构建 {out_w}x{out_h} (FOV: {fov_x_deg}°x{fov_y_deg}°) 的等距柱状映射表...")
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

    map_x = points_2d[..., 0].astype(np.float32)
    map_y = points_2d[..., 1].astype(np.float32)
    return map_x, map_y

# --- 5. 初始化摄像头 ---
cap = cv2.VideoCapture(CAMERA_ID)
ret, frame = cap.read()
if not ret:
    print("❌ 无法打开摄像头")
    sys.exit()

h, w = frame.shape[:2]

# --- 6. 配置全景展开参数 ---
OUT_W = 1280
OUT_H = 900
FOV_X = 180.0 
FOV_Y = 120.0 

map1, map2 = build_equirectangular_map(K, D, OUT_W, OUT_H, FOV_X, FOV_Y)

print("\n--- 📷 YOLO-Pose + 鱼眼等距柱状投影 已启动 ---")
print("👉 按 's' 键 : 保存当前的检测结果画面")
print("👉 按 'q' 或 'ESC' 键 : 退出程序\n")

img_count = 0

# --- 7. 实时处理循环 ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. 生成右侧的展开全景图 (这是喂给 YOLO 的完美“平坦”数据)
    panorama_frame = cv2.remap(frame, map1, map2, 
                               interpolation=cv2.INTER_LINEAR, 
                               borderMode=cv2.BORDER_CONSTANT)

    # ==========================================
    # 2. 【核心新增】将展开图送入 YOLO 进行推理
    # ==========================================
    # conf=0.5 表示只显示置信度大于 50% 的结果，你可以根据实际情况调高或调低
    results = model(panorama_frame, conf=0.5, verbose=False) 
    
    # 获取 YOLO 自动画好框和 4 个关键点的画面
    annotated_panorama = results[0].plot()

    # 3. 为了完美左右拼接，将左侧原图高度等比例缩放至 OUT_H
    orig_aspect_ratio = w / h
    new_orig_w = int(OUT_H * orig_aspect_ratio)
    resized_orig = cv2.resize(frame, (new_orig_w, OUT_H))

    # 4. 左右水平堆叠拼接图 (hstack) -> 注意这里用的是画好关键点的 annotated_panorama
    combined_frame = np.hstack((resized_orig, annotated_panorama))

    # 5. 在拼接图上打上文字标签
    cv2.putText(combined_frame, f"Saved: {img_count}", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined_frame, "Original", (10, OUT_H - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(combined_frame, "YOLO-Pose (Equirectangular)", (new_orig_w + 10, OUT_H - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 6. 缩小预览窗口的尺寸，以免撑爆屏幕
    DISPLAY_SCALE = 0.7  
    preview_w = int(combined_frame.shape[1] * DISPLAY_SCALE)
    preview_h = int(combined_frame.shape[0] * DISPLAY_SCALE)
    preview_frame = cv2.resize(combined_frame, (preview_w, preview_h))

    # 显示画面
    cv2.imshow('Fisheye YOLO-Pose Detection', preview_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        save_name = os.path.join(SAVE_DIR, f"pose_detect_{img_count:04d}.jpg")
        # 按 S 键时，保存带有 YOLO 关键点和框的高清展开图
        cv2.imwrite(save_name, annotated_panorama)  
        print(f"📸 保存成功: {save_name}")
        img_count += 1
    elif key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()