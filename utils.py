
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


# collection of utility functions

#################
# general utils #
#################

# placeholder

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