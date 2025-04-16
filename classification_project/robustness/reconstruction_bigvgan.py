import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
from torch.nn import DataParallel
from tqdm import tqdm
import sys
sys.path.append("/hpc_stor03/sjtu_home/yuxiang.zhao/F5-TTS/src")  # 替换为 F5-TTS 的实际路径
from third_party.BigVGAN.bigvgan import BigVGAN
from meldataset import get_mel_spectrogram

# 定义路径
input_dir = "/hpc_stor03/sjtu_home/yuxiang.zhao/data/waterMarked/groundtruth/test-clean"
output_dir_bigvgan = "/hpc_stor03/sjtu_home/yuxiang.zhao/data/waterMarked/bigvgan/test-clean"
bigvgan_model_path = "/hpc_stor03/sjtu_home/yuxiang.zhao/F5-TTS/bigvgan_model/models--nvidia--bigvgan_v2_24khz_100band_256x/snapshots/c329ede9e9bbc100ddf5c91e2330a61921262370"


# 设置使用的GPU
gpu_ids = [0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 BigVGAN 模型
def load_bigvgan_from_local(local_path, device):
    model = BigVGAN.from_pretrained("nvidia/bigvgan_v2_24khz_100band_256x", use_cuda_kernel=False)
    model.remove_weight_norm()
    model = model.eval().to(device)
    return model



# 主函数
if __name__ == "__main__":
    # 加载模型
    print("加载 BigVGAN 模型...")
    bigvgan_model = load_bigvgan_from_local(bigvgan_model_path, device)
    

    
    # 将模型包装为DataParallel（多GPU支持）
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        bigvgan_model = DataParallel(bigvgan_model, device_ids=gpu_ids)

    
    # 获取所有.wav文件的路径
    print("正在扫描目录以获取文件列表...")
    file_paths = [str(p) for p in Path(input_dir).rglob("*.wav")]
    print(f"找到 {len(file_paths)} 个音频文件。")
    
    # 使用tqdm显示进度条
    for file_path in tqdm(file_paths, desc="处理音频文件", unit="file"):
        # 读取音频文件
        waveform, sr = torchaudio.load(file_path)
        waveform = waveform.to(device)
        
        # 提取 Mel 频谱并生成音频
        with torch.no_grad():
            # 提取 BigVGAN 和 Vocos 的 Mel 频谱
            mel_bigvgan = get_mel_spectrogram(waveform, bigvgan_model.h)

            
            # 使用 BigVGAN 生成音频
            audio_bigvgan = bigvgan_model(mel_bigvgan).squeeze(0).cpu()
        
        # 构建输出路径
        relative_path = os.path.relpath(file_path, input_dir)
        output_path_bigvgan = os.path.join(output_dir_bigvgan, relative_path)

        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path_bigvgan), exist_ok=True)

        
        # 保存生成的音频
        sf.write(output_path_bigvgan, audio_bigvgan.numpy().T, sr)

    
