import cv2
import numpy as np
import os
import sys

# --- 1. 配置路径与参数 ---
PARAM_PATH = '/home/gsp/00fish_eye_camera/fisheye_calib_params.npz'
SAVE_DIR = '/home/gsp/00fish_eye_camera/undistorted_photos'
CAMERA_ID = 2

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

# --- 3. 初始化摄像头 ---
cap = cv2.VideoCapture(CAMERA_ID)

# 尝试读取第一帧以获取画面的真实分辨率 (h, w)
ret, frame = cap.read()
if not ret:
    print("❌ 无法打开摄像头，请检查连接。")
    sys.exit()

h, w = frame.shape[:2]

# --- 4. 获取完美四周黑边的完整数据 (复刻 Matlab Full 视图) ---
print(f"正在为分辨率 {w}x{h} 计算完整去畸变映射表...")

# a. 复制最原始的相机内参 (保证最原汁原味的几何形状，不让 OpenCV 瞎拉伸)
new_K = K.copy()

# b. 核心操作：大幅等比缩小视场 (Scale) 以装下超长的畸变拉伸尾迹
# 👈 这是关键！
# - 之前的 1.5 缩放不够大。为了露出完美的四周黑边和蝴蝶形状，我们需要将画面更进一步缩小。
# - 建议尝试从 2.5 开始向上调。这里我直接设为 3.0。
# - 如果你还觉得画面主体太大顶着边，继续把这个数字改大 (比如 3.5 或 4.0)。
scale_ratio = 1.5
new_K[0, 0] /= scale_ratio  # 缩小 X 轴焦距
new_K[1, 1] /= scale_ratio  # 缩小 Y 轴焦距

# c. 强制光学中心对齐到画布正中央 (保证画面不偏移)
new_K[0, 2] = w / 2
new_K[1, 2] = h / 2

# d. 生成映射表
map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
)

print("\n--- 📷 实时去畸变相机已启动 ---")
print("👉 按 's' 键 : 保存当前去畸变后的画面")
print("👉 按 'q' 或 'ESC' 键 : 退出程序\n")

img_count = 0

# --- 5. 实时处理与抓拍循环 ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 查表法进行像素重映射
    undistorted_frame = cv2.remap(frame, map1, map2, 
                                  interpolation=cv2.INTER_LINEAR, 
                                  borderMode=cv2.BORDER_CONSTANT)

    # 拼接画面
    combined_frame = np.hstack((frame, undistorted_frame))
    
    # 在画面上加点提示文字
    display_frame = combined_frame.copy()
    cv2.putText(display_frame, f"Saved: {img_count}", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(display_frame, "Original", (10, h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(display_frame, "Undistorted (Full Data)", (w + 10, h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Fisheye Correction Preview', display_frame)

    # 按键监听
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        # 保存纯净的、四周带超宽黑边的完整几何形状画面
        save_name = os.path.join(SAVE_DIR, f"full_data_scaled_{img_count:04d}.jpg")
        cv2.imwrite(save_name, undistorted_frame)
        print(f"📸 咔嚓！成功保存带超宽黑边的图片: {save_name}")
        img_count += 1
    elif key == ord('q') or key == 27:
        break

# --- 6. 释放资源 ---
cap.release()
cv2.destroyAllWindows()
