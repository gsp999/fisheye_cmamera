import cv2
import numpy as np
import os
import sys
from ultralytics import YOLO

# ==========================================
# 1. 几何变换工具类 (用于提取、排序、和逆向映射)
# ==========================================
class GeometryUtils:
    @staticmethod
    def order_points(pts: np.ndarray) -> np.ndarray:
        ordered = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).flatten()
        ordered[0] = pts[np.argmin(s)]       # TL (左上)
        ordered[2] = pts[np.argmax(s)]       # BR (右下)
        ordered[1] = pts[np.argmin(diff)]    # TR (右上)
        ordered[3] = pts[np.argmax(diff)]    # BL (左下)
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

# 【核心修改】替换为你的两个模型路径
OBB_MODEL_PATH = '/home/gsp/00fish_eye_camera/best_obb.pt'   
POSE_MODEL_PATH = '/home/gsp/00fish_eye_camera/best.pt' 

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 加载“黄金参数”
try:
    with np.load(PARAM_PATH) as data:
        K = data['K']
        D = data['D']
    print("✅ 成功加载相机内参和畸变系数！")
except FileNotFoundError:
    print(f"❌ 找不到参数文件：{PARAM_PATH}")
    sys.exit()

# 加载 YOLO 模型
print("正在加载 YOLO OBB 与 Pose 级联模型 ...")
try:
    obb_model = YOLO(OBB_MODEL_PATH)
    pose_model = YOLO(POSE_MODEL_PATH)
    print("✅ 两个 YOLO 模型加载成功！")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    sys.exit()

cap = cv2.VideoCapture(CAMERA_ID)
ret, frame = cap.read()
if not ret:
    print("❌ 无法打开摄像头")
    sys.exit()

h, w = frame.shape[:2]

# ==========================================
# 3. 预计算透视去畸变映射表 (balance=0)
# ==========================================
print(f"正在计算透视去畸变映射表 (balance=0)...")
new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
    K, D, (w, h), np.eye(3), balance=0
)

# 【关键修改】：使用 CV_32FC1 格式！
# 这样 map1 存的就直接是原始的 X 坐标，map2 存的就是原始的 Y 坐标，方便后续反推。
map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K, D, np.eye(3), new_K, (w, h), cv2.CV_32FC1
) 

print("\n--- 📷 透视去畸变 + 级联检测 (OBB->Pose) 已启动 ---")
img_count = 0

# ==========================================
# 4. 实时处理循环
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. 查表法极速去畸变 (右侧画面)
    undistorted_frame = cv2.remap(frame, map1, map2, 
                                  interpolation=cv2.INTER_LINEAR, 
                                  borderMode=cv2.BORDER_CONSTANT)
    
    annotated_undistorted = undistorted_frame.copy() # 用于画图的干净画布

    # ==========================================
    # 【阶段 1】：OBB 粗定位
    # ==========================================
    obb_results = obb_model(undistorted_frame, conf=0.4, verbose=False)
    
    if len(obb_results) > 0 and obb_results[0].obb is not None:
        for obb_obj in obb_results[0].obb:
            obb_params = obb_obj.xywhr[0].cpu().numpy()
            
            # 膨胀边界并排序
            obb_points = GeometryUtils.get_dilated_box_points(obb_params, pad_ratio=1.2)
            ordered_obb_points = GeometryUtils.order_points(obb_points)

            # (可视化) 画出蓝色 OBB
            cv2.polylines(annotated_undistorted, [ordered_obb_points.astype(np.int32)], True, (255, 0, 0), 2)

            # ==========================================
            # 【提取】：拉平裁剪为 256x256 正视图
            # ==========================================
            warp_size = (256, 256)
            warped_img, M = GeometryUtils.warp_image(undistorted_frame, ordered_obb_points, warp_size)

            # ==========================================
            # 【阶段 2】：Pose 找角点
            # ==========================================
            pose_results = pose_model(warped_img, conf=0.5, verbose=False)

            if len(pose_results) > 0 and pose_results[0].keypoints is not None and len(pose_results[0].keypoints) > 0:
                kpts_local = pose_results[0].keypoints.xy[0].cpu().numpy()
                
                if len(kpts_local) == 4:
                    order_idx = GeometryUtils.order_points_indices(kpts_local[:, :2])
                    kpts_local = kpts_local[order_idx]

                    # 【逆向映射 1】：256x256小图坐标 -> 右侧的去畸变图坐标
                    rect_kpts = GeometryUtils.map_points_back(kpts_local, M)

                    for i in range(4):
                        pt1 = tuple(rect_kpts[i].astype(int))
                        pt2 = tuple(rect_kpts[(i + 1) % 4].astype(int))
                        # (可视化) 连线和打点
                        cv2.line(annotated_undistorted, pt1, pt2, (0, 0, 255), 2)
                        cv2.circle(annotated_undistorted, pt1, 6, (0, 255, 255), -1)

                    # ==========================================
                    # 【逆向映射 2】：右侧去畸变图坐标 -> 左侧原始鱼眼图坐标
                    # ==========================================
                    for pt in rect_kpts:
                        x_rect, y_rect = int(pt[0]), int(pt[1])
                        
                        # 确保不越出右图边界
                        if 0 <= x_rect < w and 0 <= y_rect < h:
                            # 查表，问 map1 和 map2 它原来在哪
                            x_orig = int(map1[y_rect, x_rect])
                            y_orig = int(map2[y_rect, x_rect])
                            
                            # 确保不越出左图边界
                            if 0 <= x_orig < w and 0 <= y_orig < h:
                                # 在左侧原图上画点
                                cv2.circle(frame, (x_orig, y_orig), 6, (0, 0, 255), -1)
                                cv2.circle(frame, (x_orig, y_orig), 8, (0, 255, 0), 2)

    # 左右拼接
    combined_frame = np.hstack((frame, annotated_undistorted))

    # 打标签
    cv2.putText(combined_frame, f"Saved: {img_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined_frame, "Original Fisheye", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(combined_frame, "Rectilinear Cascade (balance=0)", (w + 10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 缩放防爆屏
    DISPLAY_SCALE = 0.8  
    preview_w = int(combined_frame.shape[1] * DISPLAY_SCALE)
    preview_h = int(combined_frame.shape[0] * DISPLAY_SCALE)
    preview_frame = cv2.resize(combined_frame, (preview_w, preview_h))

    cv2.imshow('Fisheye Rectilinear Cascade', preview_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        save_name = os.path.join(SAVE_DIR, f"rect_cascade_{img_count:04d}.jpg")
        cv2.imwrite(save_name, combined_frame)
        print(f"📸 保存成功: {save_name}")
        img_count += 1
    elif key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()