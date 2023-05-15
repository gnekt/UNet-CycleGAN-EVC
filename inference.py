# load packages
import random
import yaml
from munch import Munch
import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
import random 
from pickle import load
import os
from model import UNet
import os.path as osp

# Do NOT TOUCH
config = yaml.safe_load(open("config.yml"))
MEL_PARAMS = config.get('preprocess_params', {})

### Get configuration
log_dir = config['log_dir']

MODEL = "UNet-CycleGAN-EVC-Happy"

np.random.seed(1)
random.seed(1)
mean = torch.load("dataset/dataset_mean.pt")
std = torch.load("dataset/dataset_std.pt")

MELSPEC_PARAMS = {
    "n_mels": MEL_PARAMS["spect_params"]["n_mel_band"],
    "n_fft": MEL_PARAMS["spect_params"]["n_fft"],
    "win_length": MEL_PARAMS["spect_params"]["win_length"],
    "hop_length": MEL_PARAMS["spect_params"]["hop_length"]
}

to_melspec = torchaudio.transforms.MelSpectrogram(**MELSPEC_PARAMS)
DEVICE = "cpu"

generator = UNet()
# Load checkpoint, if exists
if os.path.exists(osp.join(f"./Models/{MODEL}", 'backup.pth')):
    print("Found checkpoint.")
    checkpoint = torch.load(osp.join(f"./Models/{MODEL}", 'backup.pth'), map_location=DEVICE) # Fix from https://github.com/pytorch/pytorch/issues/2830#issuecomment-718816292
    generator.load_state_dict(checkpoint["generators"]["n_to_e"])

def load_data(wav_path: str) -> torch.Tensor:
        """Produce mel-spectrogram given a wav file
        Args:
            wav_path (str): Wav path of the source file
        Returns:
            (MelBand, T_Mel): Mel-Spectrogram of the wav file
        """
        wave_tensor = generate_wav_tensor(wav_path)
        tensor = to_melspec(wave_tensor)
        scaled_tensor = (torch.log(1e-5 + tensor) - (mean)) / std
        tensor = torch.zeros((80, 192))
        tensor[:, :scaled_tensor.shape[1]] = scaled_tensor[:,:192]
        return torch.FloatTensor(tensor)
    
def generate_wav_tensor(wave_path: str) -> torch.Tensor:
    """Private methods that trasform a wav file into a tensor
    Args:
        wave_path (str): path of the source wav file
    Returns:
        (samples,1): tensorial representation of source wav
    """
    try:
        wave, sr = librosa.load(wave_path, sr=MEL_PARAMS["sr"])
        wave_tensor = torch.from_numpy(wave).float()
    except Exception:
        print("ds")
    return wave_tensor

in_datta = load_data("../StarGANv2-EmotionalVC/dataset/eng/ESD/0011/Neutral/0011_000228.wav")

generator.eval()
generator.to(DEVICE)
out = generator(in_datta.unsqueeze(0).unsqueeze(0).to(DEVICE))

import sys
import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from attrdict import AttrDict
from Vocoder.LibriTTS.hifi.vocoder import Generator
import librosa
import numpy as np
import torchaudio

h = None

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

cp_g = scan_checkpoint("Vocoder/LibriTTS/", 'g_')

config_file = os.path.join(os.path.split(cp_g)[0], 'config.json')
with open(config_file) as f:
    data = f.read()
json_config = json.loads(data)
h = AttrDict(json_config)

device = torch.device("cpu")
generator = Generator(h).to(device)

state_dict_g = load_checkpoint(cp_g, device)
generator.load_state_dict(state_dict_g['generator'])
generator.eval()
generator.remove_weight_norm()


with torch.no_grad():
        c = out[0,0,:,:]
        y_out = generator(c)
        y_out = y_out.view(-1).cpu()
            
print("storing sample..")
sf.write(f'{MODEL}_GEN.wav', y_out, 24000)


with torch.no_grad():
        c = in_datta.to("cpu")
        y_out = generator(c)
        y_out = y_out.view(-1).cpu()
            
print("storing sample..")
sf.write(f'{MODEL}_ORIGINAL.wav', y_out, 24000)
