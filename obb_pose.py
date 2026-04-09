import cv2
import numpy as np
import os
import sys
from ultralytics import YOLO

# ==========================================
# 1. 几何变换工具类 (从 inference.py 引入)
# ==========================================
class GeometryUtils:
    @staticmethod
    def order_points(pts: np.ndarray) -> np.ndarray:
        ordered = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).flatten()
        ordered[0] = pts[np.argmin(s)]       # TL
        ordered[2] = pts[np.argmax(s)]       # BR
        ordered[1] = pts[np.argmin(diff)]    # TR
        ordered[3] = pts[np.argmax(diff)]    # BL
        return ordered

    @staticmethod
    def order_points_indices(pts: np.ndarray) -> np.ndarray:
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).flatten()
        tl = int(np.argmin(s))
        br = int(np.argmax(s))
        tr = int(np.argmin(diff))
        bl = int(np.argmax(diff))
        return np.array([tl, tr, br, bl], dtype=np.int64)
    
    @staticmethod
    def get_dilated_box_points(obb: np.ndarray, pad_ratio: float = 1.2) -> np.ndarray:
        cx, cy, w, h, angle_rad = obb
        w_dilated = w * pad_ratio
        h_dilated = h * pad_ratio
        angle_deg = np.degrees(angle_rad)
        rect = ((cx, cy), (w_dilated, h_dilated), angle_deg)
        box_points = cv2.boxPoints(rect)
        return box_points.astype(np.float32)
    
    @staticmethod
    def warp_image(img: np.ndarray, src_pts: np.ndarray, dst_size: tuple) -> tuple:
        dst_w, dst_h = dst_size
        dst_pts = np.array([[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_img = cv2.warpPerspective(img, M, dst_size, flags=cv2.INTER_LINEAR)
        return warped_img, M
    
    @staticmethod
    def map_points_back(local_points: np.ndarray, M: np.ndarray) -> np.ndarray:
        M_inv = np.linalg.inv(M)
        points_homogeneous = np.hstack([local_points[:, :2], np.ones((local_points.shape[0], 1))])
        transformed = points_homogeneous @ M_inv.T
        w = transformed[:, 2:3]
        original_points = transformed[:, :2] / (w + 1e-8)
        return original_points.astype(np.float32)

# ==========================================
# 2. 配置路径与参数
# ==========================================
PARAM_PATH = '/home/gsp/00fish_eye_camera/fisheye_calib_params.npz'
SAVE_DIR = '/home/gsp/00fish_eye_camera/cascade_detect_results'
CAMERA_ID = 2

# 【核心修改】提供两个模型的路径
OBB_MODEL_PATH = '/home/gsp/00fish_eye_camera/best_obb.pt'   # 替换为你的 OBB 模型路径
POSE_MODEL_PATH = '/home/gsp/00fish_eye_camera/best.pt' # 替换为你的 Pose 模型路径

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 加载相机内参
try:
    with np.load(PARAM_PATH) as data:
        K = data['K']
        D = data['D']
except FileNotFoundError:
    print(f"❌ 找不到参数文件：{PARAM_PATH}")
    sys.exit()

# 加载两个模型
print("正在加载 YOLO 级联模型 ...")
try:
    obb_model = YOLO(OBB_MODEL_PATH)
    pose_model = YOLO(POSE_MODEL_PATH)
    print("✅ 两个 YOLO 模型加载成功！")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    sys.exit()

# ==========================================
# 3. 等距柱状投影映射函数
# ==========================================
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

# ==========================================
# 4. 初始化与主循环
# ==========================================
cap = cv2.VideoCapture(CAMERA_ID)
ret, frame = cap.read()
if not ret:
    print("❌ 无法打开摄像头")
    sys.exit()

h, w = frame.shape[:2]
OUT_W, OUT_H = 1280, 900
FOV_X, FOV_Y = 180.0, 120.0 

map1, map2 = build_equirectangular_map(K, D, OUT_W, OUT_H, FOV_X, FOV_Y)

print("\n--- 📷 级联检测 (OBB -> Pose) 已启动 ---")
img_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. 投影为全景图
    panorama_frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    annotated_panorama = panorama_frame.copy() # 用于画框和打点的画布

    # ==========================================
    # 【阶段 1】：使用 OBB 模型寻找目标的大致区域
    # ==========================================
    obb_results = obb_model(panorama_frame, conf=0.4, verbose=False)
    
    if len(obb_results) > 0 and obb_results[0].obb is not None:
        # 遍历画面中检测到的所有 OBB 目标
        for obb_obj in obb_results[0].obb:
            obb_params = obb_obj.xywhr[0].cpu().numpy()  # [cx, cy, w, h, angle]
            
            # 获取膨胀后的 4 个角点并排序
            obb_points = GeometryUtils.get_dilated_box_points(obb_params, pad_ratio=1.2)
            ordered_obb_points = GeometryUtils.order_points(obb_points)

            # (可视化) 在全景图上画出蓝色的 OBB 边框
            cv2.polylines(annotated_panorama, [ordered_obb_points.astype(np.int32)], True, (255, 0, 0), 2)
            cv2.putText(annotated_panorama, "OBB", tuple(ordered_obb_points[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # ==========================================
            # 【提取与变换】：将 OBB 区域抠出并拉平成 256x256 的正视图
            # ==========================================
            warp_size = (256, 256)
            warped_img, M = GeometryUtils.warp_image(panorama_frame, ordered_obb_points, warp_size)

            # ==========================================
            # 【阶段 2】：用 Pose 模型在抠出的正视图中找精确角点
            # ==========================================
            pose_results = pose_model(warped_img, conf=0.5, verbose=False)

            if len(pose_results) > 0 and pose_results[0].keypoints is not None and len(pose_results[0].keypoints) > 0:
                # 获取在 256x256 局部图中的坐标
                kpts_local = pose_results[0].keypoints.xy[0].cpu().numpy()
                
                if len(kpts_local) == 4:
                    # 重新排序为 TL, TR, BR, BL 保证连线正确
                    order_idx = GeometryUtils.order_points_indices(kpts_local[:, :2])
                    kpts_local = kpts_local[order_idx]

                    # 【核心逆向映射 1】：把局部图坐标映射回 -> 全景图坐标
                    pano_kpts = GeometryUtils.map_points_back(kpts_local, M)

                    # (可视化) 在全景图上画出 Pose 关键点和连线 (红色)
                    for i in range(4):
                        pt1 = tuple(pano_kpts[i].astype(int))
                        pt2 = tuple(pano_kpts[(i + 1) % 4].astype(int))
                        cv2.line(annotated_panorama, pt1, pt2, (0, 0, 255), 2)
                        cv2.circle(annotated_panorama, pt1, 6, (0, 255, 255), -1) # 黄色圆点

                    # 【核心逆向映射 2】：把全景图坐标查表映射回 -> 原始鱼眼图坐标
                    for pt in pano_kpts:
                        x_pano, y_pano = int(pt[0]), int(pt[1])
                        # 确保不越界
                        if 0 <= x_pano < OUT_W and 0 <= y_pano < OUT_H:
                            x_orig = int(map1[y_pano, x_pano])
                            y_orig = int(map2[y_pano, x_pano])
                            
                            if 0 <= x_orig < w and 0 <= y_orig < h:
                                # 在左侧鱼眼原图上画点 (绿色光环+红心)
                                cv2.circle(frame, (x_orig, y_orig), 6, (0, 0, 255), -1)
                                cv2.circle(frame, (x_orig, y_orig), 8, (0, 255, 0), 2)

    # 左右拼接显示逻辑
    orig_aspect_ratio = w / h
    new_orig_w = int(OUT_H * orig_aspect_ratio)
    resized_orig = cv2.resize(frame, (new_orig_w, OUT_H))

    combined_frame = np.hstack((resized_orig, annotated_panorama))

    # 打上标签
    cv2.putText(combined_frame, f"Saved: {img_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined_frame, "Fisheye Source", (10, OUT_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(combined_frame, "Cascade (OBB -> Crop -> Pose)", (new_orig_w + 10, OUT_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 缩放预览
    DISPLAY_SCALE = 0.7  
    preview_frame = cv2.resize(combined_frame, (int(combined_frame.shape[1] * DISPLAY_SCALE), int(combined_frame.shape[0] * DISPLAY_SCALE)))

    cv2.imshow('Fisheye Cascade Detection', preview_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        save_name = os.path.join(SAVE_DIR, f"cascade_{img_count:04d}.jpg")
        cv2.imwrite(save_name, combined_frame)  
        print(f"📸 保存成功: {save_name}")
        img_count += 1
    elif key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()