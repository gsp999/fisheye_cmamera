#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cascaded 2-Stage Rigid Object Detection System

This module implements a high-precision detection pipeline for finding the exact 4 corner points
of rigid rectangular objects (e.g., license plates, industrial parts) using:
- Stage 1: YOLOv8-OBB for coarse localization
- Stage 2: YOLOv8-Pose for fine corner point regression

Author: AI Assistant
Date: 2024
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path
import sys

try:
    from ultralytics import YOLO
    from ultralytics.utils import colorstr
except ImportError:
    print("错误: 请先安装 ultralytics 库")
    print("安装命令: pip install ultralytics")
    sys.exit(1)


class GeometryUtils:
    """
    Utility class for geometric transformations and point ordering.
    
    IMPORTANT: Point Order Convention
    ==================================
    Throughout this entire system, we use a consistent point ordering:
    
    Index 0: Top-Left (TL)      Index 1: Top-Right (TR)
         0 ────────────────── 1
         │                    │
         │                    │
         │                    │
         3 ────────────────── 2
    Index 3: Bottom-Left (BL)  Index 2: Bottom-Right (BR)
    
    This order is maintained in:
    - order_points() output
    - warp_image() source points
    - Pose model training labels (should match this order)
    - Final detection output
    """
    
    @staticmethod
    def order_points(pts: np.ndarray) -> np.ndarray:
        """
        Sort 4 points to standard order: [TL, TR, BR, BL] (indices 0, 1, 2, 3).
        
        Algorithm:
        1. Find the point with smallest sum (top-left) and largest sum (bottom-right)
        2. Find the point with smallest difference (top-right) and largest difference (bottom-left)
        
        Args:
            pts: Array of shape (4, 2) containing 4 (x, y) points in arbitrary order
            
        Returns:
            Ordered array of shape (4, 2) with points in order: [TL, TR, BR, BL]
            Index 0: Top-Left, Index 1: Top-Right, Index 2: Bottom-Right, Index 3: Bottom-Left
        """
        if pts.shape != (4, 2):
            raise ValueError(f"Expected 4 points with shape (4, 2), got {pts.shape}")
        
        # Initialize ordered points
        ordered = np.zeros((4, 2), dtype=np.float32)
        
        # Sum and difference along axis=1
        s = pts.sum(axis=1)  # (4,)
        diff = np.diff(pts, axis=1).flatten()  # (4,)
        
        # Top-left: smallest sum
        ordered[0] = pts[np.argmin(s)]
        # Bottom-right: largest sum
        ordered[2] = pts[np.argmax(s)]
        # Top-right: smallest difference
        ordered[1] = pts[np.argmin(diff)]
        # Bottom-left: largest difference
        ordered[3] = pts[np.argmax(diff)]
        
        return ordered

    @staticmethod
    def order_points_indices(pts: np.ndarray) -> np.ndarray:
        """
        Return indices that would reorder 4 points into [TL, TR, BR, BL].

        This is useful when you want to reorder (x, y, conf/vis) together.

        Args:
            pts: Array of shape (4, 2) containing 4 (x, y) points

        Returns:
            idx: Array of shape (4,) with indices into pts in order [TL, TR, BR, BL]
        """
        if pts.shape != (4, 2):
            raise ValueError(f"Expected 4 points with shape (4, 2), got {pts.shape}")

        s = pts.sum(axis=1)  # (4,)
        diff = np.diff(pts, axis=1).flatten()  # (4,) where diff = y - x

        tl = int(np.argmin(s))
        br = int(np.argmax(s))
        tr = int(np.argmin(diff))
        bl = int(np.argmax(diff))
        return np.array([tl, tr, br, bl], dtype=np.int64)
    
    @staticmethod
    def get_dilated_box_points(obb: np.ndarray, pad_ratio: float = 1.2) -> np.ndarray:
        """
        Convert OBB (Oriented Bounding Box) to 4 dilated corner points.
        
        The OBB format from ultralytics is: [center_x, center_y, width, height, angle]
        where angle is in RADIANS (counter-clockwise from horizontal).
        
        Args:
            obb: Array of shape (5,) containing [cx, cy, w, h, angle_rad] in image coordinates
            pad_ratio: Scaling factor to dilate the box (default: 1.2 means 20% larger)
            
        Returns:
            Array of shape (4, 2) containing 4 corner points in image coordinates
        """
        if obb.shape != (5,):
            raise ValueError(f"Expected OBB with shape (5,), got {obb.shape}")
        
        cx, cy, w, h, angle_rad = obb
        
        # Dilate width and height
        w_dilated = w * pad_ratio
        h_dilated = h * pad_ratio
        
        # Convert angle from radians to degrees for OpenCV
        angle_deg = np.degrees(angle_rad)
        
        # Create rotated rectangle
        rect = ((cx, cy), (w_dilated, h_dilated), angle_deg)
        
        # Get 4 corner points using OpenCV
        box_points = cv2.boxPoints(rect)  # Returns (4, 2) array
        
        return box_points.astype(np.float32)
    
    @staticmethod
    def warp_image(
        img: np.ndarray, 
        src_pts: np.ndarray, 
        dst_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply perspective transform to warp image from source points to destination rectangle.
        
        Mathematical Background:
        The perspective transform matrix M maps points from source to destination:
            [x']   [m00 m01 m02] [x]
            [y'] = [m10 m11 m12] [y]
            [w ]   [m20 m21  1 ] [1]
        
        where (x', y') = (x'/w, y'/w) are the transformed coordinates.
        
        Args:
            img: Input image (H, W, C)
            src_pts: Source points array of shape (4, 2) - should be ordered [tl, tr, br, bl]
            dst_size: Destination size as (width, height) tuple
            
        Returns:
            warped_img: Warped image of size dst_size
            M: Perspective transform matrix of shape (3, 3)
        """
        if src_pts.shape != (4, 2):
            raise ValueError(f"Expected 4 source points with shape (4, 2), got {src_pts.shape}")
        
        dst_w, dst_h = dst_size
        
        # Define destination points (canonical rectangle)
        dst_pts = np.array([
            [0, 0],           # top-left
            [dst_w - 1, 0],  # top-right
            [dst_w - 1, dst_h - 1],  # bottom-right
            [0, dst_h - 1]   # bottom-left
        ], dtype=np.float32)
        
        # Compute perspective transform matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Warp image
        warped_img = cv2.warpPerspective(img, M, dst_size, flags=cv2.INTER_LINEAR)
        
        return warped_img, M
    
    @staticmethod
    def map_points_back(local_points: np.ndarray, M: np.ndarray) -> np.ndarray:
        """
        Map points from warped/local coordinate system back to original image coordinates.
        
        Mathematical Background:
        We use the inverse transform M_inv = M^(-1) to map points back:
            [x_orig]   [m00' m01' m02'] [x_local]
            [y_orig] = [m10' m11' m12'] [y_local]
            [w      ]   [m20' m21'  1 ] [1      ]
        
        Args:
            local_points: Points in warped coordinate system, shape (N, 2) or (N, 3)
            M: Perspective transform matrix of shape (3, 3)
            
        Returns:
            Points in original image coordinates, shape (N, 2)
        """
        if M.shape != (3, 3):
            raise ValueError(f"Expected transform matrix with shape (3, 3), got {M.shape}")
        
        # Compute inverse transform
        M_inv = np.linalg.inv(M)
        
        # Handle both 2D and 3D point formats
        if local_points.shape[1] == 2:
            # Add homogeneous coordinate (w=1)
            points_homogeneous = np.hstack([local_points, np.ones((local_points.shape[0], 1))])
        elif local_points.shape[1] == 3:
            # Already has visibility/confidence, use only x, y
            points_homogeneous = np.hstack([local_points[:, :2], np.ones((local_points.shape[0], 1))])
        else:
            raise ValueError(f"Expected points with 2 or 3 columns, got {local_points.shape[1]}")
        
        # Transform points: [N, 3] @ [3, 3]^T = [N, 3]
        transformed = points_homogeneous @ M_inv.T
        
        # Convert from homogeneous to Cartesian coordinates
        w = transformed[:, 2:3]  # (N, 1)
        original_points = transformed[:, :2] / (w + 1e-8)  # (N, 2)
        
        return original_points.astype(np.float32)


class CascadeDetector:
    """
    Cascaded 2-stage detector for rigid object corner point detection.
    
    Pipeline:
    1. Stage 1 (OBB): Detect oriented bounding box
    2. Geometric Rectification: Dilate OBB, compute perspective transform
    3. Stage 2 (Pose): Detect 4 corner keypoints in warped image
    4. Inverse Mapping: Map keypoints back to original image coordinates
    """
    
    def __init__(
        self,
        obb_model_path: str,
        pose_model_path: str,
        pad_ratio: float = 1.2,
        warp_size: Tuple[int, int] = (256, 256),
        conf_threshold: float = 0.25,
        device: str = ""
    ):
        """
        Initialize the cascaded detector.
        
        Args:
            obb_model_path: Path to YOLOv8-OBB model (.pt file)
            pose_model_path: Path to YOLOv8-Pose model (.pt file)
            pad_ratio: Dilation factor for OBB (default: 1.2 = 20% larger)
            warp_size: Size of warped image for Stage 2 (width, height)
            conf_threshold: Confidence threshold for detections
            device: Device to run inference on ('cpu', '0', '1', etc.)
        """
        self.pad_ratio = pad_ratio
        self.warp_size = warp_size
        self.conf_threshold = conf_threshold
        
        # Load models
        print(f"{colorstr('green', 'bold', '→')} 正在加载 Stage 1 (OBB) 模型: {obb_model_path}")
        self.obb_model = YOLO(obb_model_path)
        
        print(f"{colorstr('green', 'bold', '→')} 正在加载 Stage 2 (Pose) 模型: {pose_model_path}")
        self.pose_model = YOLO(pose_model_path)
        
        if device:
            self.obb_model.to(device)
            self.pose_model.to(device)
        
        print(f"{colorstr('green', 'bold', '✓')} 模型加载完成")
        print(f"  膨胀比例: {pad_ratio}")
        print(f"  变换尺寸: {warp_size}")
        print(f"  置信度阈值: {conf_threshold}")
    
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Main prediction pipeline.
        
        Args:
            image: Input image as numpy array (H, W, C) in BGR format
            
        Returns:
            Dictionary containing:
                - 'obb': OBB detection result (None if no detection)
                - 'obb_points': 4 dilated corner points from OBB (None if no detection)
                - 'warped_image': Warped image for Stage 2 (None if no detection)
                - 'keypoints': 4 corner keypoints in original image coordinates (None if no detection)
                - 'keypoints_local': 4 corner keypoints in warped image coordinates (None if no detection)
                - 'transform_matrix': Perspective transform matrix M (None if no detection)
                - 'success': Boolean indicating if detection was successful
        """
        result = {
            'obb': None,
            'obb_points': None,
            'warped_image': None,
            'keypoints': None,
            'keypoints_local': None,
            'transform_matrix': None,
            'success': False
        }
        
        # ========== Stage 1: OBB Detection ==========
        obb_results = self.obb_model.predict(
            image,
            conf=self.conf_threshold,
            verbose=False
        )
        
        if len(obb_results) == 0 or obb_results[0].obb is None or len(obb_results[0].obb) == 0:
            print(f"{colorstr('yellow', 'bold', '⚠')} Stage 1: 未检测到目标")
            return result
        
        # Get first detection (assuming single object)
        obb_obj = obb_results[0].obb
        result['obb'] = obb_obj
        
        # Extract OBB parameters: [cx, cy, w, h, angle]
        # xywhr returns shape (N, 5) for N detections
        obb_params = obb_obj.xywhr[0].cpu().numpy()  # Shape: (5,)
        
        # ========== Geometric Rectification ==========
        # Get dilated corner points
        obb_points = GeometryUtils.get_dilated_box_points(obb_params, self.pad_ratio)
        result['obb_points'] = obb_points
        
        # Order points: [tl, tr, br, bl]
        ordered_obb_points = GeometryUtils.order_points(obb_points)
        
        # Warp image
        warped_img, M = GeometryUtils.warp_image(image, ordered_obb_points, self.warp_size)
        result['warped_image'] = warped_img
        result['transform_matrix'] = M
        
        # ========== Stage 2: Pose Detection ==========
        pose_results = self.pose_model.predict(
            warped_img,
            conf=self.conf_threshold,
            verbose=False
        )
        
        if len(pose_results) == 0 or len(pose_results[0].keypoints) == 0:
            print(f"{colorstr('yellow', 'bold', '⚠')} Stage 2: 未检测到关键点")
            return result
        
        # Get keypoints from first detection
        keypoints_obj = pose_results[0].keypoints[0]
        keypoints_local = keypoints_obj.xy[0].cpu().numpy()  # Shape: (4, 2) or (4, 3)
        
        # Ensure we have exactly 4 keypoints
        if keypoints_local.shape[0] != 4:
            print(f"{colorstr('yellow', 'bold', '⚠')} Stage 2: 检测到 {keypoints_local.shape[0]} 个关键点，期望 4 个")
            return result

        # Reorder keypoints by geometry into [TL, TR, BR, BL] to make downstream usage stable
        xy = keypoints_local[:, :2].astype(np.float32)
        order_idx = GeometryUtils.order_points_indices(xy)
        keypoints_local = keypoints_local[order_idx]
        result['keypoints_local'] = keypoints_local
        
        # ========== Inverse Mapping ==========
        keypoints_original = GeometryUtils.map_points_back(keypoints_local, M)
        result['keypoints'] = keypoints_original
        
        result['success'] = True
        return result
    
    def visualize(
        self,
        image: np.ndarray,
        result: Dict[str, Any],
        show_obb: bool = True,
        show_keypoints: bool = True,
        point_labels: bool = True
    ) -> np.ndarray:
        """
        Visualize detection results on image.
        
        Point labeling follows this convention:
            0(TL) ────── 1(TR)
             │            │
             │            │
            3(BL) ────── 2(BR)
        
        Args:
            image: Original input image
            result: Result dictionary from predict() method
            show_obb: Whether to draw OBB (Stage 1) in blue
            show_keypoints: Whether to draw keypoints (Stage 2) in red
            point_labels: Whether to label keypoints with indices and names
            
        Returns:
            Annotated image
        """
        vis_image = image.copy()
        
        # Define corner names for better visualization
        corner_names = ['TL', 'TR', 'BR', 'BL']  # Top-Left, Top-Right, Bottom-Right, Bottom-Left
        
        # Draw OBB (Stage 1) in Blue
        if show_obb and result['obb_points'] is not None:
            obb_points = result['obb_points'].astype(np.int32)
            cv2.polylines(vis_image, [obb_points], isClosed=True, color=(255, 0, 0), thickness=2)
            cv2.putText(
                vis_image, "OBB (Stage 1)", 
                tuple(obb_points[0].astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
            )
        
        # Draw Keypoints (Stage 2) in Red
        if show_keypoints and result['keypoints'] is not None:
            keypoints = result['keypoints'].astype(np.int32)
            
            # Draw points with different colors for each corner
            colors = [(0, 255, 0), (0, 255, 255), (0, 0, 255), (255, 0, 255)]  # Green, Yellow, Red, Magenta
            
            for i, pt in enumerate(keypoints):
                color = colors[i] if i < len(colors) else (0, 0, 255)
                cv2.circle(vis_image, tuple(pt), radius=6, color=color, thickness=-1)
                cv2.circle(vis_image, tuple(pt), radius=8, color=(255, 255, 255), thickness=2)  # White border
                
                if point_labels:
                    # Label with both index and corner name
                    label = f"{i}({corner_names[i]})"
                    cv2.putText(
                        vis_image, label,
                        tuple(pt + np.array([12, -12])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                    )
            
            # Draw connections: 0->1->2->3->0
            if len(keypoints) == 4:
                for i in range(4):
                    pt1 = tuple(keypoints[i])
                    pt2 = tuple(keypoints[(i + 1) % 4])
                    cv2.line(vis_image, pt1, pt2, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            
            # Add legend
            cv2.putText(
                vis_image, "Keypoints Order: 0(TL) -> 1(TR) -> 2(BR) -> 3(BL)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
        
        return vis_image


def main():
    """Example usage of the cascaded detector."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Cascaded 2-Stage Rigid Object Detection')
    parser.add_argument('--obb-model', type=str, required=True,
                       help='Path to YOLOv8-OBB model (.pt file)')
    parser.add_argument('--pose-model', type=str, required=True,
                       help='Path to YOLOv8-Pose model (.pt file)')
    parser.add_argument('--source', type=str, required=True,
                       help='Input image path or camera index (0, 1, ...)')
    parser.add_argument('--pad-ratio', type=float, default=1.3,
                       help='Dilation ratio for OBB (default: 1.2)')
    parser.add_argument('--warp-size', type=str, default='256,256',
                       help='Warped image size as width,height (default: 256,256)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--device', type=str, default='',
                       help='Device to run inference on (default: auto)')
    parser.add_argument('--save', action='store_true',
                       help='Save output image')
    parser.add_argument('--show', action='store_true',
                       help='Show result image')
    
    args = parser.parse_args()
    
    # Parse warp size
    warp_size = tuple(map(int, args.warp_size.split(',')))
    
    # Initialize detector
    detector = CascadeDetector(
        obb_model_path=args.obb_model,
        pose_model_path=args.pose_model,
        pad_ratio=args.pad_ratio,
        warp_size=warp_size,
        conf_threshold=args.conf,
        device=args.device
    )
    
    # Load image or use camera
    if args.source.isdigit():
        # Camera input
        cap = cv2.VideoCapture(int(args.source))
        if not cap.isOpened():
            print(f"错误: 无法打开摄像头 {args.source}")
            return
        
        print("按 'q' 键退出摄像头")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            result = detector.predict(frame)
            
            # Visualize
            vis_frame = detector.visualize(frame, result)
            
            # Show
            cv2.imshow('Cascade Detection', vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    else:
        # Image input
        image = cv2.imread(args.source)
        if image is None:
            print(f"错误: 无法加载图像 {args.source}")
            return
        
        # Run detection
        print(f"\n{colorstr('green', 'bold', '→')} 开始检测...")
        result = detector.predict(image)
        
        if result['success']:
            print(f"{colorstr('green', 'bold', '✓')} 检测成功!")
            obb_params = result['obb'].xywhr[0].cpu().numpy()
            angle_deg = np.degrees(obb_params[4])  # Convert radians to degrees for display
            print(f"  OBB 参数: [cx={obb_params[0]:.2f}, cy={obb_params[1]:.2f}, "
                  f"w={obb_params[2]:.2f}, h={obb_params[3]:.2f}, angle={angle_deg:.2f}°]")
            print(f"\n  关键点顺序说明: 0(TL-左上) -> 1(TR-右上) -> 2(BR-右下) -> 3(BL-左下)")
            print(f"  关键点坐标 (原始图像):")
            corner_names = ['TL(左上)', 'TR(右上)', 'BR(右下)', 'BL(左下)']
            for i, pt in enumerate(result['keypoints']):
                print(f"    Point {i} {corner_names[i]:8s}: ({pt[0]:7.2f}, {pt[1]:7.2f})")
        else:
            print(f"{colorstr('yellow', 'bold', '⚠')} 检测失败")
        
        # Visualize
        vis_image = detector.visualize(image, result)
        
        # Show
        if args.show:
            cv2.imshow('Cascade Detection', vis_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Save
        if args.save:
            output_path = Path(args.source).stem + '_result.jpg'
            cv2.imwrite(output_path, vis_image)
            print(f"{colorstr('green', 'bold', '✓')} 结果已保存到: {output_path}")


if __name__ == '__main__':
    main()
