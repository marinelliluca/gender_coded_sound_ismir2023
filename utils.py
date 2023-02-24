
import warnings
import librosa
import torch
import torch.nn as nn
from music.backend import Backend
from music.frontend import Frontend

# collection of utility functions

#######################
# pytorch model utils #
#######################

# compute mel-spectrogram for the MSD model
def compute_melspectrogram(audio_fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x, _ = librosa.core.load(audio_fn, sr=16000, res_type='kaiser_fast')
        spec = librosa.core.amplitude_to_db(librosa.feature.melspectrogram(x, # type: ignore
                                                                           sr=16000, 
                                                                           n_fft=512, 
                                                                           hop_length=256, 
                                                                           n_mels=96))
    return spec

# load pytorch model parameters 
def load_parameters(model, filename): 
    model = nn.DataParallel(model)
    S = torch.load(filename)
    model.load_state_dict(S)
    return model

# assemble model
class AssembleModel(nn.Module):
    
    def __init__(self, main_dict):
        super(AssembleModel, self).__init__()

        self.frontend = Frontend(main_dict)
        self.backend = Backend(main_dict)


    def forward(self, spec):
        
        x = self.backend(self.frontend(spec))
        
        return x