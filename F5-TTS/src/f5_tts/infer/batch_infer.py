import os
import subprocess
from datetime import datetime
import threading


# Set inference script path
infer_script = "/PATH/TO/F5-TTS/src/f5_tts/infer/infer_cli.py"
# Set input and output paths
input_dir = "/PATH/TO/DATA"
output_base_dir = "/PATH/TO/OUTPUT"

# Static parameters
model = "F5-TTS"  # F5-TTS | E2-TTS

# GPU configuration
gpus = [0,1,2,3,4,5,6,7]  # GPU indices to use
gpu_count = len(gpus)
thread_limit = 2 # Max threads per GPU
threads = []

# Define inference thread function
def run_inference(gen_text, ref_audio, ref_text, output_dir, selected_gpu, sample_id):
    command = [
        "python3", infer_script,
        "--model", model,
        "--ref_audio", ref_audio, # Reference audio path
        "--gen_text", gen_text,  # Generated text path
        "--ref_text", ref_text,  # Reference text path
        "--output_dir", output_dir,   # Output directory
        "--name", sample_id,          # Audio sample ID
        "--vocoder_name","bigvgan",   # Vocoder
        #"--load_vocoder_from_local",  # Load vocoder model locally
        "-p", "/PATH/TO/F5-TTS/ckpts/F5TTS_Base/model_1200000.pt" # Pretrained model path
        
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)

    # Call inference script
    start_time = datetime.now()
    process = subprocess.Popen(command, env=env)
    process.wait()
    end_time = datetime.now()
    print(f"[{end_time}] Generation complete: ID {sample_id}, GPU: {selected_gpu}, Time: {end_time - start_time}")

existing_file_count = 0
missing_ref_file_count = 0
# Iterate through all files in input directory
for root, dirs, files in os.walk(input_dir):
    for file_name in files:
        if file_name.endswith(".normalized.txt"):
            # Process text file path
            txt_path = os.path.join(root, file_name)
            gen_text = open(txt_path, "r", encoding="utf-8").read().strip()
            # print(text)
            # Find .wav file with different name than current txt file
            base_name = file_name.replace(".normalized.txt", "")
            ref_audio = None
            for audio_file in os.listdir(root):
                if audio_file.endswith(".wav") and audio_file != f"{base_name}.wav":
                    ref_audio = os.path.join(root, audio_file)
                    break  # Exit after finding first .wav file with different name

            if not ref_audio:
                missing_ref_file_count += 1
                print(f"Warning: No ref_audio file found for {txt_path}, skipping.")
                continue

            # Find .normalized.txt file with same name as ref_audio
            ref_text_path = ref_audio.replace(".wav", ".normalized.txt")
            if not os.path.exists(ref_text_path):
                missing_ref_file_count += 1
                print(f"Warning: No ref_text file found for {ref_audio}, skipping.")
                continue

            ref_text = open(ref_text_path, "r", encoding="utf-8").read().strip()

            relative_path = os.path.relpath(root, input_dir)
            output_dir = os.path.join(output_base_dir, relative_path)
            os.makedirs(output_dir, exist_ok=True)

            sample_id = base_name
            output_wav_path = os.path.join(output_dir, f"{sample_id}.wav")
            '''During GAN training, do not skip generation, overwrite previous audio files'''
            if os.path.exists(output_wav_path):
                existing_file_count += 1
                print(f"File exists, skipping generation: {output_wav_path}")
                continue
            selected_gpu = gpus[len(threads) % gpu_count]
            thread = threading.Thread(target=run_inference, args=(gen_text, ref_audio, ref_text, output_dir, selected_gpu, sample_id))
            threads.append(thread)
            thread.start()

            # Check if parallel task limit is reached
            if len(threads) >= thread_limit * gpu_count:
                # Wait for all threads to complete before continuing
                for t in threads:
                    t.join()
                threads.clear()  # Clear thread list
# Wait for remaining threads to complete
for t in threads:
    t.join()
print(f"Existing files count: {existing_file_count}")
print(f"Missing ref files count: {missing_ref_file_count}")