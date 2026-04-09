import subprocess
import os

def convert_webm_to_mp4(input_file: str, output_file: str):
    """
    将 WEBM 视频转换为 MP4 视频
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 '{input_file}'")
        return

    # 构建 FFmpeg 执行命令
    # 使用 libx264 视频编码和 aac 音频编码以确保最佳的 MP4 兼容性
    command = [
        'ffmpeg',
        '-i', input_file,      
        '-c:v', 'libx264',     
        '-c:a', 'aac',         
        '-y',                  # 如果输出文件已存在，则自动覆盖 (可选)
        output_file            
    ]

    try:
        print(f"正在转换: {input_file} -> {output_file} ...")
        # 执行命令，如果转换失败则抛出异常
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print("✅ 转换成功！")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 转换过程中发生错误。返回码: {e.returncode}")
    except FileNotFoundError:
        print("❌ 错误: 未找到 FFmpeg。请确保已安装 FFmpeg 并已添加到系统环境变量中。")

# 使用示例
if __name__ == "__main__":
    # 将此处的路径替换为你自己的文件路径
    input_video = "/home/gsp/视频/录屏/1.webm"
    output_video = "result.mp4"
    
    convert_webm_to_mp4(input_video, output_video)