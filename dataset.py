#coding: utf-8

import yaml
from typing import Tuple, Dict, List
import os
import random
import random
import torch
import torchaudio
import pedalboard
import numpy as np
import soundfile as sf
import torch.nn.functional as F

from torch import nn, Tensor
from torch.utils.data import DataLoader
import pandas as pd
import logging
import librosa
import librosa.display
from pickle import load

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# Do NOT TOUCH
config = yaml.safe_load(open("config.yml"))
MEL_PARAMS = config.get('preprocess_params', {})

np.random.seed(1)
random.seed(1)

MELSPEC_PARAMS = {
    "n_mels": MEL_PARAMS["spect_params"]["n_mel_band"],
    "n_fft": MEL_PARAMS["spect_params"]["n_fft"],
    "win_length": MEL_PARAMS["spect_params"]["win_length"],
    "hop_length": MEL_PARAMS["spect_params"]["hop_length"]
}

mean = torch.load("dataset/dataset_mean.pt")
std = torch.load("dataset/dataset_std.pt")
###########################################################

class Dataset(torch.utils.data.Dataset):
    """Dataset container
    Args:
        Dataset: extend base torch class Dataset
    """

    def __init__(self,
                 dataset: pd.DataFrame,
                 validation: bool = False
                 ):
        """Constructor
        Args:
            dataset (pd.DataFrame): Data.
            validation (bool, optional): If the dataset is in Validation mode. Defaults to False.
        """
        self.dataset = dataset
        self.dataset["already_used"] = False
        self.validation = validation
        print(validation)
        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MELSPEC_PARAMS)
        
    def __len__(self) -> int:
        """Cardinality of the dataset
        Returns:
            (int): The cardinality
        """
        return self.dataset.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Get a sample from the set
        Args:
            idx (int): Index of the selected sample
            
        Returns:
            (
               Source Spectrogram -> (T_Mel, MelBand),
               Reference Spectrogram -> (T_Mel, MelBand)
            )
        """
        row = self.dataset.iloc[idx]
        
        neutral_random_scale = random.uniform(0.5, 1) if not self.validation else 1
        emotion_random_scale = random.uniform(0.5, 1) if not self.validation else 1
            
        neutral_tensor = self._load_data(row["source_path"], neutral_random_scale)
        emotion_tensor = self._load_data(row["reference_path"], emotion_random_scale)
        
        return neutral_tensor, emotion_tensor

    
    def _load_data(self, wav_path: str, random_scale=1) -> torch.Tensor:
        """Produce mel-spectrogram given a wav file
        Args:
            wav_path (str): Wav path of the source file
            random_scale (float, optional): Random scale to apply to the wav. Defaults to 1.
            
        Returns:
            (MelBand, T_Mel): Mel-Spectrogram of the wav file
        """
        wave_tensor = self._generate_wav_tensor(wav_path)
        
        if random_scale != 1:
            wave_tensor = wave_tensor * random_scale        
        
        wave_tensor = torch.from_numpy(wave_tensor).float()
        mel_w: Tensor = self.to_melspec(wave_tensor)
        
        mel_w: Tensor = (torch.log(1e-5 + mel_w) - mean) / std
        return torch.FloatTensor(mel_w)
    
    def _generate_wav_tensor(self, wave_path: str) -> torch.Tensor:
        """Private methods that trasform a wav file into a tensor
        Args:
            wave_path (str): path of the source wav file
        Returns:
            (samples,1): tensorial representation of source wav
        """
        try:
            wave, sr = librosa.load(wave_path, sr=MEL_PARAMS["sr"])            
        except Exception:
            print(wave_path)
        return wave


class Collater(object):

    def __init__(self, batch_size: int = 5):
        """Constructor

        Args:
            batch_size (int, optional): Nr. of sample per batch. Defaults to 5.
        """        
        self.batch_size = batch_size
        self.max_t_mel = 192

    def __call__(self, batch: List) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            batch (List): _description_

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: (
                Neutral Mel Spectrogram -> (Batch, 1, MelBand, T_Mel),
            )
        """        
        batch_size = len(batch)
        max_lenght_mel_tensor = self.max_t_mel
        
        # Create padded tensor, following the shape (batch_size, MelBand, T_Mel)
        neutral_padded_tensor = torch.full(
            (batch_size, MELSPEC_PARAMS["n_mels"], max_lenght_mel_tensor), .0)
        emotion_padded_tensor = torch.full(
            (batch_size, MELSPEC_PARAMS["n_mels"], max_lenght_mel_tensor ), .0) # Full accept number not a tensor 

        # Create padding mask, following the shape (batch_size, T_Mel as boolean)
        neutral_padding_mask = torch.full((batch_size, max_lenght_mel_tensor), False) # False = padded token
        emotion_padding_mask = torch.full((batch_size, max_lenght_mel_tensor), False) # False = padded token
        
        for bid, (neutral_tensor, emotion_tensor) in enumerate(batch):
            if neutral_tensor.shape[1] > self.max_t_mel:   # if mel has a len greater than the max allowed, trim it! 
                random_start = np.random.randint(0, neutral_tensor.shape[1] - self.max_t_mel)
                neutral_tensor = neutral_tensor[:, random_start:random_start + self.max_t_mel]
            
            if emotion_tensor.shape[1] > self.max_t_mel:   # if mel has a len greater than the max allowed, trim it! 
                random_start = np.random.randint(0, emotion_tensor.shape[1] - self.max_t_mel)
                emotion_tensor = emotion_tensor[:, random_start:random_start + self.max_t_mel]
            
            # Attach to batch and padding
            neutral_padded_tensor[bid, :, :neutral_tensor.shape[1]] = neutral_tensor
            emotion_padded_tensor[bid, :, :emotion_tensor.shape[1]] = emotion_tensor
            
            neutral_padding_mask[bid, :neutral_tensor.shape[1]] = True
            emotion_padding_mask[bid, :emotion_tensor.shape[1]] = True
        
        # Why unsqueeze? We put these tensor inside 2D CNN, that expect a 4D tensor (batch, channel, height(MelBand), width(Time))
        return neutral_padded_tensor.unsqueeze(1), emotion_padded_tensor.unsqueeze(1), neutral_padding_mask, emotion_padding_mask


def build_dataloader(dataset_path: str,
                     dataset_configuration: Dict,
                     batch_size: int = 4,
                     num_workers: int = 1,
                     device: str = 'cpu',
                     validation = False
                     ) -> DataLoader:
    """Make a dataloader
    Args:
        dataset_path (str): Path of the source dataset 
        dataset_configuration (Dict): Dataset configuration
            {
                "data_separetor": str,
                "data_header": List[str] -> List of the header column of the dataset
            }
        batch_size (int, optional): Batch Size. Defaults to 4.
        num_workers (int, optional): Number of Workers. Defaults to 1.
        device (str, optional): Device. Defaults to 'cpu'.
    Raise
        FileNotFoundError: If the data_path is not a file

    Returns:
        DataLoader: The pytorch dataloader
    """

    # Get Dataset info
    separetor = dataset_configuration["data_separetor"]
    data_header = dataset_configuration["data_header"]
    ####

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Check path! {dataset_path} does not exist!")

    dataset = pd.read_csv(dataset_path, sep=separetor, names=data_header)
    dataset = Dataset(dataset, validation)
    collate_fn = Collater(batch_size)

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=num_workers,
                             drop_last=True,
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))

    return data_loader