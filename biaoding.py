import cv2
import numpy as np
import glob

# --- 1. 参数设置 ---
CHECKERBOARD = (10, 7) # 棋盘格内角点数量 (列数, 行数)
SQUARE_SIZE = 15       # 每个方块的实际物理边长 (毫米)

# 停止迭代的标准：达到最大迭代次数 30，或者精度达到 0.1
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW

# 准备真实世界中的 3D 点云坐标
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

# 储存所有图片的 3D 点和 2D 点
objpoints = [] # 真实世界中的 3D 点
imgpoints = [] # 图像平面中的 2D 像素点

# --- 2. 提取角点 ---
images = glob.glob('/home/gsp/00fish_eye_camera/photo/*.jpg') 

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, 
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    if ret == True:
        objpoints.append(objp)
        # 亚像素级角点精细化，提高标定精度
        corners2 = cv2.cornerSubPix(gray, corners, (3,3), (-1,-1), subpix_criteria)
        imgpoints.append(corners2)

# --- 3. 进行鱼眼标定 ---
N_OK = len(objpoints)
K = np.zeros((3, 3)) # 内参矩阵
D = np.zeros((4, 1)) # 畸变系数
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)] 
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)] 

print("正在计算标定参数，请稍候...")
rms, _, _, _, _ = cv2.fisheye.calibrate(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    K,
    D,
    rvecs,
    tvecs,
    calibration_flags,
    (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
)

print(f"标定重投影误差 (RMS): {rms:.4f}")
print("内参矩阵 (K):\n", K)
print("畸变系数 (D):\n", D)

# ==========================================
# --- 4. 优先保存标定参数 ---
# ==========================================
save_path = '/home/gsp/00fish_eye_camera/fisheye_calib_params.npz'
np.savez(save_path, K=K, D=D)
print(f"🎉 标定参数已成功保存至: {save_path}")


# ==========================================
# --- 5. 图像去畸变预览 ---
# ==========================================
img_test = cv2.imread('/home/gsp/00fish_eye_camera/test.jpg')

if img_test is None:
    print("⚠️ 未找到测试图片 test.jpg，已跳过去畸变预览。")
else:
    h,  w = img_test.shape[:2]

    # 计算映射表 (balance=0.0 切除废像素区域)
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=0.0)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)

    # 执行重映射
    undistorted_img = cv2.remap(img_test, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    print("👉 正在显示对比图，看完后请在图片窗口按【键盘任意键】退出程序。")
    cv2.imshow('Original', img_test)
    cv2.imshow('Undistorted', undistorted_img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()