
import warnings
from contextlib import contextmanager
import sys, os
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset # type: ignore
from pytorch_lightning.core import LightningModule # type: ignore
from torch.nn import functional as F
from music.backend import Backend
from music.frontend import Frontend
import numpy as np
import yaml


# collection of utility functions

#################
# general utils #
#################

def load_embeddings_and_labels(groundtruth_df, emotions_and_mid_level, emb_dim, which, modality, voice, n_tar_cls):

    with open("fn_suffix.yaml", "r") as stream:
        fn_suffix = yaml.safe_load(stream)

    # load embeddings
    X = np.empty((groundtruth_df.shape[0], emb_dim))
    y_reg = np.empty((emotions_and_mid_level.shape[0], emotions_and_mid_level.shape[1]))

    for i,stimulus_id in enumerate(groundtruth_df.index):
        embedding = np.load(f"{modality}/embeddings_{which}{'' if voice else '_novoice'}/" +
                            f"{stimulus_id}{fn_suffix[modality][which]}.npy")
        X[i] = embedding.mean(axis=0)
        y_reg[i] = emotions_and_mid_level.loc[stimulus_id].values

    tar_classes = ["Girls/women", "Boys/men"] if n_tar_cls==2 else ["Girls/women", "Mixed", "Boys/men"]
    y_tar = groundtruth_df.target.values
    y_tar = [tar_classes.index(x) if x in tar_classes else -1 for x in y_tar]
    y_tar = np.array(y_tar)
    
    y_voig = None
    if voice:
        voig_classes = ['Feminine', 'Masculine', 'There are no voices', 'BOTH feminine and masculine voices']
        y_voig = groundtruth_df.voice_gender.values
        y_voig = [voig_classes.index(x) if x in voig_classes else -1 for x in y_voig]
        y_voig = np.array(y_voig)

    return X, y_reg, y_tar, y_voig

###################
# MSD model utils #
###################

# compute mel-spectrogram for the MSD model
def compute_melspectrogram(audio_fn, sr=16000, n_fft=512, hop_length=256, n_mels=96):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.core.load(audio_fn, sr=16000, res_type='kaiser_fast')
        spec = librosa.core.amplitude_to_db(librosa.feature.melspectrogram(y=y, # type: ignore
                                                                           sr=sr, 
                                                                           n_fft=n_fft, 
                                                                           hop_length=hop_length, 
                                                                           n_mels=n_mels))
    return spec

# load pytorch model parameters 
def load_parameters(model, filename): 
    model = nn.DataParallel(model) # type: ignore
    S = torch.load(filename)
    model.load_state_dict(S)
    return model

# assemble MSD model
class AssembleModel(nn.Module):
    
    def __init__(self, main_dict):
        super(AssembleModel, self).__init__()

        self.frontend = Frontend(main_dict)
        self.backend = Backend(main_dict)


    def forward(self, spec):
        
        x = self.backend(self.frontend(spec))
        
        return x
    

#######################
# class and reg utils #
#######################

class EmbeddingsDataset(Dataset):
    def __init__(self, X, y_reg, y_cls):
        self.X = torch.from_numpy(X).float()
        self.y_reg = torch.from_numpy(y_reg).float()
        self.y_cls = torch.from_numpy(y_cls).long()
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_reg[idx], self.y_cls[idx]


class MultipleRegressionWithSoftmax(LightningModule):
    def __init__(self, input_dim, n_regressions, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, n_regressions)
        self.linear3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.linear(x))
        x_reg = self.linear2(x)
        x_cls = self.linear3(x)
        return x_reg, x_cls

    def training_step(self, batch, batch_idx):
        x, y_reg, y_cls = batch
        y_hat1, y_hat2 = self(x)
        
        loss1 = nn.MSELoss()(y_hat1, y_reg)
        
        # Handle missing data in the cost function
        loss2 = F.cross_entropy(y_hat2, y_cls, ignore_index=-1)
        
        loss = loss1 + loss2
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_reg, y_cls = batch
        y_hat1, y_hat2 = self(x)
        
        loss1 = nn.MSELoss()(y_hat1, y_reg)
        
        # Handle missing data in the cost function
        loss2 = F.cross_entropy(y_hat2, y_cls, ignore_index=-1)
        
        loss = loss1 + loss2
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-5) # type: ignore


class EmbeddingsDataset2(Dataset):
    def __init__(self, X, y_reg, y_tar, y_voig):
        self.X = torch.from_numpy(X).float()
        self.y_reg = torch.from_numpy(y_reg).float()
        self.y_tar = torch.from_numpy(y_tar).long()
        self.y_voig = torch.from_numpy(y_voig).long()
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_reg[idx], self.y_tar[idx], self.y_voig[idx]


class MegaModelV1(LightningModule):
    def __init__(self, input_dim, n_regressions, output_dim1, output_dim2):
        super().__init__()
        self.linear = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, n_regressions)
        self.linear3 = nn.Linear(128, output_dim1)
        self.linear4 = nn.Linear(128, output_dim2)

    def forward(self, x):
        x = F.relu(self.linear(x))
        x_reg = self.linear2(x)
        x_tar = self.linear3(x)
        x_voig = self.linear4(x)
        return x_reg, x_tar, x_voig

    def training_step(self, batch, batch_idx):
        x, y_reg, y_tar, y_voig = batch
        y_hat1, y_hat2, y_hat3 = self(x)
        
        loss1 = nn.MSELoss()(y_hat1, y_reg)
        
        # Handle missing data in the cost function
        loss2 = F.cross_entropy(y_hat2, y_tar, ignore_index=-1)
        
        # Handle missing data in the cost function
        loss3 = F.cross_entropy(y_hat3, y_voig, ignore_index=-1)
        
        loss = loss1 + loss2 + loss3
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_reg, y_tar,y_voig  = batch
        y_hat1,y_hat2,y_hat3  = self(x)
        
        loss1 = nn.MSELoss()(y_hat1,y_reg )
        
        # Handle missing data in the cost function
        loss2 = F.cross_entropy(y_hat2,y_tar , ignore_index=-1)
        
         # Handle missing data in the cost function
        loss3 = F.cross_entropy(y_hat3,y_voig , ignore_index=-1)
        
        loss = loss1 + loss2 +loss3
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-5)
