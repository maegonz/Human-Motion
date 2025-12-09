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


# import numpy as np
# import pandas as pd
# import torch
# from os.path import join
# from transformers import AutoTokenizer, T5ForConditionalGeneration
# from torch.utils.data import Dataset, DataLoader

# # Charger le tokenizer
# MODEL_NAME = "t5-small"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# class MotionDataset(Dataset):
#     def __init__(self, descriptions, motions, tokenizer, max_length=128):
#         self.descriptions = descriptions
#         self.motions = motions
#         self.tokenizer = tokenizer
#         self.max_length = max_length
    
#     def __len__(self):
#         return len(self.descriptions)
    
#     def __getitem__(self, idx):
#         motion = torch.tensor(self.motions[idx], dtype=torch.float32)
#         encoding = self.tokenizer(
#             self.descriptions[idx],
#             padding="max_length",
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors="pt"
#         )
#         return motion, encoding.input_ids.squeeze(), encoding.attention_mask.squeeze()

# def load_data(file_list, motion_data_dir, text_data_dir):
#     data = {}
#     with open(file_list, 'r') as f:
#         titles = f.read().splitlines()
#         for t in titles[:10]:
#             npy_file = f"{t}.npy"
#             motion_data = np.load(join(motion_data_dir, npy_file))

#             # Charger et traiter la description
#             txt_file = f"{text_data_dir}{t}.txt"
#             with open(txt_file, 'r', encoding='utf-8') as m:
#                 desc = m.readline().split('#')[0].capitalize()

#             data[desc] = motion_data

#     return pd.DataFrame({'Description': list(data.keys()), 'Motion': list(data.values())})

# def pad_motion_sequences(motions):
#     max_length = max(m.shape[0] for m in motions)  # Trouver la séquence la plus longue
#     N, d = motions[0].shape[1], motions[0].shape[2]  # Garder N et d constants

#     # Initialisation d'un tenseur rempli de zéros
#     padded_motions = torch.zeros(len(motions), max_length, N, d, dtype=torch.float32)

#     # Remplissage des séquences réelles
#     for i, motion in enumerate(motions):
#         T = motion.shape[0]  # Longueur de la séquence actuelle
#         padded_motions[i, :T, :, :] = torch.tensor(motion, dtype=torch.float32)  # Copier la séquence

#     return padded_motions

# def collate_fn(batch):
#     motions, input_ids, attention_mask = zip(*batch)
#     print(type(motions))
#     print(motions[0].shape)
#     print(motions[0].shape[0])
    
#     # Padding des séquences de mouvement
#     padded_motions = pad_motion_sequences(motions)

#     input_ids = torch.stack(input_ids)
#     attention_mask = torch.stack(attention_mask)

#     return padded_motions, input_ids, attention_mask


# # Répertoires des données
# motion_data_dir = "./motions/"
# text_data_dir = "./texts/"

# # Charger les ensembles de données
# traindf = load_data('train.txt', motion_data_dir, text_data_dir)
# valdf = load_data('val.txt', motion_data_dir, text_data_dir)

# # Affichage des tailles des ensembles
# print('Train:', traindf.shape)
# print('Validation:', valdf.shape)

# xtrain, ytrain = traindf['Motion'], traindf['Description']
# xval, yval = valdf['Motion'], valdf['Description']

# print('OK')

# dataset = MotionDataset(ytrain, xtrain, tokenizer)
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# # Modèle Transformer
# model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# print('OK')

# # Entraînement (exemple d'une seule époque)
# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
# criterion = torch.nn.CrossEntropyLoss()

# for epoch in range(2):
#     for motions, input_ids, attention_mask in dataloader:
#         # print(type(motions))
#         # print(motions.shape)
#         # print(input_ids)
#         # print(input_ids.shape)
#         # print(attention_mask)
#         # print(attention_mask.shape)
#         # break
#         input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
#         optimizer.zero_grad()
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()
#     print(f"Époque {epoch+1}, Perte: {loss.item():.4f}")