import os
import subprocess
from datetime import datetime
import threading


# 设置推理文件路径
infer_script = "/hpc_stor03/sjtu_home/yuxiang.zhao/F5-TTS/src/f5_tts/infer/infer_cli.py"
# 设置输入输出路径
input_dir = "/hpc_stor03/sjtu_home/yuxiang.zhao/data/groundtruth/LibriTTS/test-clean"
output_base_dir = "/hpc_stor03/sjtu_home/yuxiang.zhao/data/MM/f5tts/origin/test-clean"

# 静态参数
model = "F5-TTS"  # F5-TTS | E2-TTS

# GPU 配置
gpus = [0,1,2,3]  # 使用的GPU编号
gpu_count = len(gpus)
thread_limit = 2 # 每张卡最多运行4个线程
threads = []

# 定义推理线程的函数
def run_inference(gen_text, ref_audio, ref_text, output_dir, selected_gpu, sample_id):
    command = [
        "python3", infer_script,
        "--model", model,
        "--ref_audio", ref_audio,# 参考音频路径
        "--gen_text", gen_text,  # 生成文本路径
        "--ref_text", ref_text,  # 参考文本路径
        "--output_dir", output_dir,   # 输出路径
        "--name", sample_id,          # 音频编号
        "--vocoder_name","vocos",   # 声码器
        "--load_vocoder_from_local",  # 本地加载声码器模型
        "-p", "/hpc_stor03/sjtu_home/yuxiang.zhao/F5-TTS/ckpts/F5TTS_Base/model_1200000.pt" # 加载预训练模型路径
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)

    # 调用推理脚本
    start_time = datetime.now()
    process = subprocess.Popen(command, env=env)
    process.wait()
    end_time = datetime.now()
    print(f"[{end_time}] 完成生成: 编号 {sample_id}, GPU: {selected_gpu}, 用时: {end_time - start_time}")

cnt = 0
cc = 0
# 遍历输入目录的所有文件
for root, dirs, files in os.walk(input_dir):
    for file_name in files:
        if file_name.endswith(".normalized.txt"):
            # 处理 text 文件路径
            txt_path = os.path.join(root, file_name)
            gen_text = open(txt_path, "r", encoding="utf-8").read().strip()
            # print(text)
            # 查找与当前 txt 文件不同名的 .wav 文件
            base_name = file_name.replace(".normalized.txt", "")
            ref_audio = None
            for audio_file in os.listdir(root):
                if audio_file.endswith(".wav") and audio_file != f"{base_name}.wav":
                    ref_audio = os.path.join(root, audio_file)
                    break  # 找到第一个不同名的 .wav 文件后退出

            if not ref_audio:
                cc += 1
                print(f"警告：未找到与 {txt_path} 对应的 ref_audio 文件，跳过此文件。")
                continue

            # 查找与 ref_audio 同名的 .normalized.txt 文件
            ref_text_path = ref_audio.replace(".wav", ".normalized.txt")
            if not os.path.exists(ref_text_path):
                cc += 1
                print(f"警告：未找到与 {ref_audio} 对应的 ref_text 文件，跳过此文件。")
                continue

            ref_text = open(ref_text_path, "r", encoding="utf-8").read().strip()

            relative_path = os.path.relpath(root, input_dir)
            output_dir = os.path.join(output_base_dir, relative_path)
            os.makedirs(output_dir, exist_ok=True)

            sample_id = base_name
            output_wav_path = os.path.join(output_dir, f"{sample_id}.wav")
            '''gan训练中,不跳过生成,覆盖上一轮的音频文件
            if os.path.exists(output_wav_path):
                cnt += 1
                print(f"文件已存在，跳过生成：{output_wav_path}")
                continue'''
            selected_gpu = gpus[len(threads) % gpu_count]
            # audio_test = "/hpc_stor03/sjtu_home/yuxiang.zhao/F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_en.wav"
            # text_test = "Some call me nature, others call me mother nature."
            thread = threading.Thread(target=run_inference, args=(gen_text, ref_audio, ref_text, output_dir, selected_gpu, sample_id))
            threads.append(thread)
            thread.start()

            # 检查是否达到并行任务限制
            if len(threads) >= thread_limit * gpu_count:
                # 等待所有线程完成后继续
                for t in threads:
                    t.join()
                threads.clear()  # 清空线程列表
# 等待剩余线程完成
for t in threads:
    t.join()
print(f"已存在的文件数：{cnt}")
print(f"未找到 ref 文件数：{cc}")