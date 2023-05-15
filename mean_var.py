import pandas as pd
import numpy as np
import random
import yaml
import torch
import soundfile as sf
import torchaudio
import tqdm
from pickle import dump

######################################## Do NOT TOUCH
config = yaml.safe_load(open("Configs/config.yml"))
MEL_PARAMS = config.get('preprocess_params', {})
np.random.seed(1)
random.seed(1)

MEL_PARAMS = {
    "n_mels": MEL_PARAMS["spect_params"]["n_mel_band"],
    "n_fft": MEL_PARAMS["spect_params"]["n_fft"],
    "win_length": MEL_PARAMS["spect_params"]["win_length"],
    "hop_length": MEL_PARAMS["spect_params"]["hop_length"]
}
to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)
###########################################################

def _generate_wav_tensor(wave_path: str) -> torch.Tensor:
        """Private methods that trasform a wav file into a tensor
        Args:
            wave_path (str): path of the source wav file
        Returns:
            (samples,1): tensorial representation of source wav
        """
        try:
            wave, sr = sf.read(wave_path)
            wave_tensor = torch.from_numpy(wave).float()
        except Exception:
            print("ds")
        return wave_tensor


def load_data(wav_path: str) -> torch.Tensor:
        """Produce mel-spectrogram given a wav file
        Args:
            wav_path (str): Wav path of the source file
        Returns:
            (MelBand, T_Mel): Mel-Spectrogram of the wav file
        """
        wave_tensor = _generate_wav_tensor(wav_path)
        
        return torch.log(1e-5 + to_melspec(wave_tensor))
    
# Define stream 
dataframe = pd.read_csv("dataset/dataset.csv", sep=";").sample(frac=1)

mel_band = torch.Tensor()

for index,row in tqdm.tqdm(dataframe[:8000].iterrows()):
    tensor = load_data(row["path"]).transpose(1, 0)
    mel_band = torch.cat((mel_band, tensor)) 

std, mean = torch.std_mean(mel_band,0)

torch.save(std,"dataset/dataset_std.pt")
torch.save(mean,"dataset/dataset_mean.pt")