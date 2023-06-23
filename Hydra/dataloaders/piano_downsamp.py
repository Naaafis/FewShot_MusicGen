import os
import random
import torch
from scipy.io.wavfile import read
from torch import Tensor
from scipy.signal import resample

from pathlib import Path
import numpy as np
from torchvision import datasets, models, transforms

from torch.utils.data.distributed import DistributedSampler
from scipy.io.wavfile import read as wavread

from typing import Tuple

import torchaudio
from torch.utils.data import Dataset

def files_to_list(data_path):
    files = [os.path.join(data_path, f.rstrip()) for f in os.listdir(data_path) if len(f)>=4 and f[-4:]=='.wav']
    return files

def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.from_numpy(data).float(), sampling_rate

def fix_length(tensor, length):
    assert len(tensor.shape) == 2 and tensor.shape[0] == 1
    if tensor.shape[1] > length:
        return tensor[:,:length]
    elif tensor.shape[1] < length:
        return torch.cat([tensor, torch.zeros(1, length-tensor.shape[1])], dim=1)
    else:
        return tensor

def float32_to_int16(waveform):
    waveform *= 32767
    waveform = waveform.short()
    return waveform

def load_piano_samples_item(filepath: str, path: str, segment_length: int, sampling_rate: int):
    relpath = os.path.relpath(filepath, path)
    filename = os.path.split(relpath)[1]
    
    # Load audio
    waveform, loaded_sample_rate = torchaudio.load(filepath)

    # Ensure audio is no longer than 1 second and is fixed length
    waveform = fix_length(waveform, segment_length)
    
    # Downsize sampling rate to target
    waveform = torchaudio.transforms.Resample(loaded_sample_rate, sampling_rate)(waveform)

    # Convert float32 waveform to int16
    waveform = float32_to_int16(waveform)
    
    return (waveform, sampling_rate, filename)

class PianoSamples(Dataset):
    """
    Create a Dataset for Piano Samples. Each item is a tuple of the form:
    waveform, sample_rate, filename
    """

    def __init__(self, path: str, segment_length: int, sampling_rate: int):
        self._path = path
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate
        self._walker = sorted(str(p) for p in Path(self._path).glob('**/*.wav'))

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str]:
        fileid = self._walker[n]
        return load_piano_samples_item(fileid, self._path, self.segment_length, self.sampling_rate)

    def __len__(self) -> int:
        return len(self._walker)
