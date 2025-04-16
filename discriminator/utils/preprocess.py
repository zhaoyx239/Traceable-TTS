# utils/preprocess.py
import torchaudio

def load_audio(file_path, sr=16000):
    waveform, sample_rate = torchaudio.load(file_path,backend="ffmpeg")
    if sample_rate != sr:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sr)(waveform)
    return waveform
