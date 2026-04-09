import cv2
import numpy as np
import os
import sys
from ultralytics import YOLO

# ==========================================
# 1. 全局配置参数
# ==========================================
PARAM_PATH = '/home/gsp/00fish_eye_camera/fisheye_calib_params.npz'
SAVE_DIR = '/home/gsp/00fish_eye_camera/cascade_detect_results'
CAMERA_ID = 2

# 替换为你的级联模型路径
OBB_MODEL_PATH = '/home/gsp/00fish_eye_camera/best_obb.pt'   
POSE_MODEL_PATH = '/home/gsp/00fish_eye_camera/best.pt' 

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


# ==========================================
# 2. 几何与级联检测核心引擎 (底层工具)
# ==========================================
class GeometryUtils:
    @staticmethod
    def order_points(pts):
        ordered = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).flatten()
        ordered[0] = pts[np.argmin(s)]       # TL
        ordered[2] = pts[np.argmax(s)]       # BR
        ordered[1] = pts[np.argmin(diff)]    # TR
        ordered[3] = pts[np.argmax(diff)]    # BL
        return ordered

    @staticmethod
    def order_points_indices(pts):
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).flatten()
        tl, br = int(np.argmin(s)), int(np.argmax(s))
        tr, bl = int(np.argmin(diff)), int(np.argmax(diff))
        return np.array([tl, tr, br, bl], dtype=np.int64)
    
    @staticmethod
    def get_dilated_box_points(obb, pad_ratio=1.2):
        cx, cy, w, h, angle_rad = obb
        angle_deg = np.degrees(angle_rad)
        rect = ((cx, cy), (w * pad_ratio, h * pad_ratio), angle_deg)
        return cv2.boxPoints(rect).astype(np.float32)
    
    @staticmethod
    def warp_image(img, src_pts, dst_size):
        dst_w, dst_h = dst_size
        dst_pts = np.array([[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return cv2.warpPerspective(img, M, dst_size, flags=cv2.INTER_LINEAR), M
    
    @staticmethod
    def map_points_back(local_points, M):
        M_inv = np.linalg.inv(M)
        points_homo = np.hstack([local_points[:, :2], np.ones((local_points.shape[0], 1))])
        transformed = points_homo @ M_inv.T
        return (transformed[:, :2] / (transformed[:, 2:3] + 1e-8)).astype(np.float32)

class CascadeDetector:
    """级联检测处理器：负责运行 OBB -> 抠图 -> Pose -> 坐标双向反推"""
    @staticmethod
    def process(target_img, original_img, map1, map2, obb_model, pose_model):
        annotated_target = target_img.copy()
        annotated_orig = original_img.copy() if original_img is not None else None
        
        # 1. 运行 OBB
        obb_results = obb_model(target_img, conf=0.4, verbose=False)
        if len(obb_results) > 0 and obb_results[0].obb is not None:
            for obb_obj in obb_results[0].obb:
                obb_params = obb_obj.xywhr[0].cpu().numpy()
                obb_points = GeometryUtils.get_dilated_box_points(obb_params, pad_ratio=1.2)
                ordered_obb = GeometryUtils.order_points(obb_points)
                
                # 画框
                cv2.polylines(annotated_target, [ordered_obb.astype(np.int32)], True, (255, 0, 0), 2)

                # 2. 抠图放大
                warped_img, M = GeometryUtils.warp_image(target_img, ordered_obb, (256, 256))
                
                # 3. 运行 Pose
                pose_results = pose_model(warped_img, conf=0.5, verbose=False)
                if len(pose_results) > 0 and pose_results[0].keypoints is not None and len(pose_results[0].keypoints) > 0:
                    kpts = pose_results[0].keypoints.xy[0].cpu().numpy()
                    if len(kpts) == 4:
                        kpts = kpts[GeometryUtils.order_points_indices(kpts[:, :2])]
                        
                        # 4. 反推到去畸变图
                        target_kpts = GeometryUtils.map_points_back(kpts, M)
                        for i in range(4):
                            pt1, pt2 = tuple(target_kpts[i].astype(int)), tuple(target_kpts[(i+1)%4].astype(int))
                            cv2.line(annotated_target, pt1, pt2, (0, 0, 255), 2)
                            cv2.circle(annotated_target, pt1, 6, (0, 255, 255), -1)

                        # 5. 反推到原始鱼眼图 (如果提供了 original_img 和映射表)
                        if annotated_orig is not None and map1 is not None and map2 is not None:
                            h_t, w_t = target_img.shape[:2]
                            h_o, w_o = annotated_orig.shape[:2]
                            for pt in target_kpts:
                                x_t, y_t = int(pt[0]), int(pt[1])
                                if 0 <= x_t < w_t and 0 <= y_t < h_t:
                                    x_o, y_o = int(map1[y_t, x_t]), int(map2[y_t, x_t])
                                    if 0 <= x_o < w_o and 0 <= y_o < h_o:
                                        cv2.circle(annotated_orig, (x_o, y_o), 6, (0, 0, 255), -1)
                                        cv2.circle(annotated_orig, (x_o, y_o), 8, (0, 255, 0), 2)
                                        
        return annotated_target, annotated_orig


# ==========================================
# 3. 投影策略类 (三种方案)
# ==========================================

class CubemapProjection:
    """方案三：立方体投影 (六面十字架展开) - 整图识别版"""
    def __init__(self, K, D, img_w, img_h, face_size=400):
        self.face_size = face_size
        print(f"初始化 立方体投影 (面分辨率 {face_size}x{face_size})...")
        
        f = c = face_size / 2.0
        K_new = np.array([[f, 0, c], [0, f, c], [0, 0, 1]], dtype=np.float32)
        
        rotations = {
            'front':  np.linalg.inv(np.array([[ 1,0,0], [0,1,0], [0,0,1]])),
            'right':  np.linalg.inv(np.array([[ 0,0,1], [0,1,0], [-1,0,0]])),
            'left':   np.linalg.inv(np.array([[ 0,0,-1],[0,1,0], [ 1,0,0]])),
            'top':    np.linalg.inv(np.array([[ 1,0,0], [0,0,-1],[ 0,1,0]])),
            'bottom': np.linalg.inv(np.array([[ 1,0,0], [0,0,1], [ 0,-1,0]])),
            'back':   np.linalg.inv(np.array([[-1,0,0], [0,1,0], [ 0,0,-1]]))
        }
        
        self.maps = {}
        for name, R in rotations.items():
            self.maps[name] = cv2.fisheye.initUndistortRectifyMap(K, D, R, K_new, (face_size, face_size), cv2.CV_32FC1)

    def process_frame(self, frame, obb_model, pose_model):
        faces = {}
        # 1. 提取 6 个面 (先不识别)
        for name in ['front', 'right', 'left', 'top', 'bottom', 'back']:
            faces[name] = cv2.remap(frame, self.maps[name][0], self.maps[name][1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        # 2. 拼装成一整张十字架底图
        s = self.face_size
        cross_bg = np.zeros((s * 3, s * 4, 3), dtype=np.uint8)
        
        cross_bg[0:s, s:s*2] = faces['top']
        cross_bg[s:s*2, 0:s] = faces['left']
        cross_bg[s:s*2, s:s*2] = faces['front']
        cross_bg[s:s*2, s*2:s*3] = faces['right']
        cross_bg[s:s*2, s*3:s*4] = faces['back']
        cross_bg[s*2:s*3, s:s*2] = faces['bottom']
        
        # 3. 【核心修改】将拼好的整张大图，作为一个整体送给 YOLO！
        # 注：因为整张图包含了 6 个不同的空间坐标系，这里的反推原图功能传 None (关闭)
        ann_cross, _ = CascadeDetector.process(cross_bg, None, None, None, obb_model, pose_model)
        
        # 打个标签
        cv2.putText(ann_cross, "Whole Cubemap Recognition", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        return ann_cross


# ==========================================
# 4. 主程序入口
# ==========================================
if __name__ == "__main__":
    # 1. 加载参数与模型
    with np.load(PARAM_PATH) as data:
        K, D = data['K'], data['D']
    
    obb_model = YOLO(OBB_MODEL_PATH)
    pose_model = YOLO(POSE_MODEL_PATH)
    
    cap = cv2.VideoCapture(CAMERA_ID)
    ret, frame = cap.read()
    if not ret: sys.exit("无法打开摄像头")
    h, w = frame.shape[:2]

    # ==============================================================
    # 🌟🌟🌟 核心开关：在这里切换你想要的投影方式！ 🌟🌟🌟
    # ==============================================================
    
    # 方案 A: 等距柱状投影 (全景图)
    # projector = EquirectangularProjection(K, D, w, h)
    
    # 方案 B: 透视投影 (传统去畸变)
    # projector = RectilinearProjection(K, D, w, h, balance=0.0)
    
    # 方案 C: 立方体投影 (十字切片)
    projector = CubemapProjection(K, D, w, h, face_size=400)
    
    # ==============================================================

    print(f"\n--- 📷 统一视觉架构启动 [{projector.__class__.__name__}] ---")
    img_count = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 调用对应类的处理函数 (内置了重映射 + YOLO检测 + 画图拼图)
        result_frame = projector.process_frame(frame, obb_model, pose_model)

        # 缩放预览以防爆屏
        DISPLAY_SCALE = 0.6
        preview = cv2.resize(result_frame, (int(result_frame.shape[1] * DISPLAY_SCALE), int(result_frame.shape[0] * DISPLAY_SCALE)))
        
        cv2.imshow('Unified Vision System', preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            save_name = os.path.join(SAVE_DIR, f"{projector.__class__.__name__}_{img_count:04d}.jpg")
            cv2.imwrite(save_name, result_frame)
            print(f"📸 战况保存成功: {save_name}")
            img_count += 1
        elif key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()