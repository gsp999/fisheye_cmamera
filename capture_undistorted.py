import cv2
import numpy as np
import os
import sys
from ultralytics import YOLO  # 【新增】导入 YOLO 库

# --- 1. 配置路径与参数 ---
PARAM_PATH = '/home/gsp/00fish_eye_camera/fisheye_calib_params.npz'
SAVE_DIR = '/home/gsp/00fish_eye_camera/yolo_rectilinear_photos'  # 换了个保存目录名以示区分
CAMERA_ID = 2

# 【新增】你的 YOLO-Pose 模型路径
YOLO_MODEL_PATH = '/home/gsp/00fish_eye_camera/best.pt' 

# 如果保存目录不存在，自动创建
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# --- 2. 加载“黄金参数” ---
try:
    with np.load(PARAM_PATH) as data:
        K = data['K']
        D = data['D']
    print("✅ 成功加载相机内参和畸变系数！")
except FileNotFoundError:
    print(f"❌ 找不到参数文件：{PARAM_PATH}")
    print("请确认你已经在之前的标定代码末尾添加了保存 .npz 文件的代码并运行过。")
    sys.exit()

# --- 3. 初始化摄像头与 YOLO 模型 ---
print(f"正在加载 YOLO 模型: {YOLO_MODEL_PATH} ...")
try:
    model = YOLO(YOLO_MODEL_PATH)
    print("✅ YOLO 模型加载成功！")
except Exception as e:
    print(f"❌ YOLO 模型加载失败，请检查路径是否正确。\n错误信息: {e}")
    sys.exit()

cap = cv2.VideoCapture(CAMERA_ID)

# 尝试读取第一帧以获取画面的真实分辨率 (h, w)
ret, frame = cap.read()
if not ret:
    print("❌ 无法打开摄像头，请检查连接。")
    sys.exit()

h, w = frame.shape[:2]

# --- 4. 核心优化：预计算映射表 (只需执行一次) ---
print(f"正在为分辨率 {w}x{h} 计算透视去畸变映射表 (balance=1)...")

# balance 设为 1，保留所有像素，但左右可能会被严重拉伸
new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
    K, D, (w, h), np.eye(3), balance=0
)

map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
) 

print("\n--- 📷 透视投影 + YOLO-Pose 已启动 ---")
print("👉 按 's' 键 : 保存当前的检测结果画面")
print("👉 按 'q' 或 'ESC' 键 : 退出程序\n")

img_count = 0

# --- 5. 实时处理与抓拍循环 ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. 极速去畸变：查表法进行像素重映射 (透视投影)
    undistorted_frame = cv2.remap(frame, map1, map2, 
                                  interpolation=cv2.INTER_LINEAR, 
                                  borderMode=cv2.BORDER_CONSTANT)

    # ==========================================
    # 2. 【核心新增】将透视去畸变后的图送入 YOLO 进行推理
    # ==========================================
    results = model(undistorted_frame, conf=0.5, verbose=False) 
    
    # 获取 YOLO 画好关键点和边界框的画面
    annotated_frame = results[0].plot()

    # 3. 拼接画面，方便你直观对比 (左边原图，右边 YOLO 检测图)
    combined_frame = np.hstack((frame, annotated_frame))
    
    # 4. 在画面上加点提示文字
    display_frame = combined_frame.copy()
    cv2.putText(display_frame, f"Saved: {img_count}", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(display_frame, "Original", (10, h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(display_frame, "YOLO-Pose (Rectilinear balance=1)", (w + 10, h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # ==========================================
    # 5. 【新增】缩小预览窗口的尺寸，以免撑爆屏幕
    # ==========================================
    DISPLAY_SCALE = 0.8  # 缩小到原来的 50%，你可以根据屏幕大小改为 0.6 或 0.7
    preview_w = int(display_frame.shape[1] * DISPLAY_SCALE)
    preview_h = int(display_frame.shape[0] * DISPLAY_SCALE)
    preview_frame = cv2.resize(display_frame, (preview_w, preview_h))

    cv2.imshow('Fisheye Correction Preview', preview_frame)

    # 按键监听
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        # 保存带有 YOLO 关键点和框的高清结果图
        save_name = os.path.join(SAVE_DIR, f"rect_pose_{img_count:04d}.jpg")
        cv2.imwrite(save_name, annotated_frame)
        print(f"📸 咔嚓！成功保存检测图片: {save_name}")
        img_count += 1
    elif key == ord('q') or key == 27:
        break

# --- 6. 释放资源 ---
cap.release()
cv2.destroyAllWindows()