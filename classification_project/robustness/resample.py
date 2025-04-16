import os
import torchaudio
from torchaudio.transforms import Resample
from datetime import datetime
import threading
from tqdm import tqdm

# 输入和输出路径
input_dir = "/hpc_stor03/sjtu_home/yuxiang.zhao/data/infer/cosyvoice2/test-clean"
output_base_dir = "/hpc_stor03/sjtu_home/yuxiang.zhao/data/MM/cosyvoice2/8k/test-clean"  # 请自行修改

# 每次并行处理的线程数限制
thread_limit = 16
threads = []

# 定义重采样函数
def resample_audio(input_path, output_path, orig_freq=24000, new_freq=8000):
    """
    将音频从原始采样率转换为目标采样率，并保存到指定路径。
    """
    try:
        # 加载音频
        waveform, sample_rate = torchaudio.load(input_path)
        if False and sample_rate != orig_freq:
            tqdm.write(f"警告：文件 {input_path} 的采样率 ({sample_rate}) 与期望值 ({orig_freq}) 不匹配。")
        
        # 重采样
        resampler = Resample(orig_freq=sample_rate, new_freq=new_freq)
        resampled_waveform = resampler(waveform)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存重采样音频
        torchaudio.save(output_path, resampled_waveform, new_freq)
        tqdm.write(f"[{datetime.now()}] 完成重采样: {input_path} -> {output_path}")
    except Exception as e:
        tqdm.write(f"错误：处理文件 {input_path} 时出现问题: {e}")

# 收集所有待处理的文件
file_pairs = []  # 存储 (input_path, output_path)
for root, _, files in os.walk(input_dir):
    for file_name in files:
        if file_name.endswith(".wav"):  # 只处理 .wav 文件
            input_path = os.path.join(root, file_name)
            
            # 构造输出路径，保持相对目录结构
            relative_path = os.path.relpath(root, input_dir)
            output_dir = os.path.join(output_base_dir, relative_path)
            output_path = os.path.join(output_dir, file_name)

            # 检查是否已经存在
            if os.path.exists(output_path):
                continue
            file_pairs.append((input_path, output_path))

# 用 tqdm 展示进度条
with tqdm(total=len(file_pairs), desc="重采样进度") as pbar:
    for input_path, output_path in file_pairs:
        # 创建线程进行重采样
        thread = threading.Thread(target=resample_audio, args=(input_path, output_path))
        threads.append(thread)
        thread.start()

        # 控制并行线程数
        if len(threads) >= thread_limit:
            for t in threads:
                t.join()  # 等待当前线程完成
                pbar.update(1)  # 更新进度条
            threads.clear()  # 清空线程列表

    # 等待剩余线程完成
    for t in threads:
        t.join()
        pbar.update(1)  # 更新进度条
