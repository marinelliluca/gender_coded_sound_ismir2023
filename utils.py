
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

#######################
# MSD model utils #
#######################

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
    

########################
# classification utils #
########################

class EmbeddingsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class SoftmaxClassifier(LightningModule):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.drop = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = self.bn(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-5) # type: ignore
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)
        return loss

class MultipleRegression(LightningModule):
    def __init__(self, input_dim, n_regressions):
        super().__init__()
        self.linear = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, n_regressions)
    
    def forward(self, x):
        x = F.relu(self.linear(x))
        x = self.linear2(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=5e-5) # type: ignore