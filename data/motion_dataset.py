import numpy as np
import pandas as pd
import torch
from os.path import join
from typing import Optional, Union
from Path import Path
from torch.utils.data import Dataset, DataLoader


class MotionDataset(Dataset):
    def __init__(self,
                 text_dir: Union[str, Path],
                 motion_dir: Union[str, Path]):
        """
        Params
        -------
        text_dir : str, Path
            Directory with all the text descriptions files.
        motion_dir : str, Path
            Directory with all motion files.
        """

        # path of the text descripions and motions directory
        # .../text/
        # .../motions/
        self.text_dir = text_dir
        self.motion_dir = motion_dir

        self.split_dict = self._split()
    
    def __len__(self):
        return True
    
    def __getitem__(self, idx):
        motion = torch.tensor(self.motions[idx], dtype=torch.float32)
        text = True
        return {
            "motion": motion,
            "text": text
        }

    def _get_motion(self,):
        return True
    
    def _get_description(self,):
        return True
    
    def _load_motion(self, name):
        npy_file = f"{name}.npy"
        motion_data = np.load(join(self.motion_dir, npy_file))
        return motion_data
    
    def _split():
        split_dict = {}

        for file in Path('./ground_truth/').iterdir():
            with open(file, 'r') as f:
                names = f.read().splitlines()
                split_dict[f.name] = names
        
        return split_dict