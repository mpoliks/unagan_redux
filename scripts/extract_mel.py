import os
import librosa
import numpy as np
import warnings
from pathlib import Path
from joblib import Parallel, delayed

import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

class Audio2Mel(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        sampling_rate=44100,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=None,
    ):
        super().__init__()
        ##############################################
        # FFT Parameters                              #
        ##############################################
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax

        window = torch.hann_window(win_length).float()
        mel_basis = librosa.filters.mel(
            sr=self.sampling_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mel_channels,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)

    def forward(self, audio):
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (p, p), "reflect").squeeze(1)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
        )
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        return log_mel_spec

def is_valid_data(data):
    return not (np.isnan(data).any() or np.isinf(data).any())

def convert_file(extract_func, sampling_rate, path):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="PySoundFile failed. Trying audioread instead."
        )
        y, _ = librosa.load(path, sr=sampling_rate)

    peak = np.abs(y).max()
    if peak_norm or peak > 1.0:
        y /= peak

    y = torch.from_numpy(y)
    y = y[None, None]
    mel = extract_func(y)
    mel = mel.numpy()
    mel = mel[0]
    # print(mel.shape)

    return mel.astype(np.float32)

def process_clip(extract_func, sampling_rate, base_out_dir, clip_path):
    id = Path(clip_path).stem

    out_dir = base_out_dir / "mel"
    out_dir.mkdir(exist_ok=True, parents=True)

    out_fp = out_dir / f"{id}.npy"

    if out_fp.exists():
        print(f"Clip {out_fp} exists. Done before.")
        return

    mel = convert_file(extract_func, sampling_rate, clip_path)
    if is_valid_data(mel):
        np.save(out_fp, mel, allow_pickle=False)
    else:
        print(f"Skipping invalid data in file: {clip_path}")

if __name__ == "__main__":
    base_out_dir = Path("./training_data/exp_data/")
    clip_dir = Path("./training_data/clips/")  # out_dir from step1

    feat_type = "mel"
    extension = ".mp3"
    peak_norm = True

    n_fft = 1024
    hop_length = 256
    win_length = 1024
    n_mel_channels = 80
    sampling_rate = 22050

    # ### Process ###
    extract_func = Audio2Mel(
        n_fft, hop_length, win_length, sampling_rate, n_mel_channels
    )

    clip_paths = sorted(clip_dir.glob("*.mp3"))
    Parallel(n_jobs=-1, verbose=2, pre_dispatch="all")(
        delayed(process_clip)(extract_func, sampling_rate, base_out_dir, clip_path)
        for clip_path in clip_paths
    )
