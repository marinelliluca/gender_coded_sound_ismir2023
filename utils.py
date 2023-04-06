import warnings
from contextlib import contextmanager
import sys, os
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset  # type: ignore
from pytorch_lightning.core import LightningModule  # type: ignore
from torch.nn import functional as F
from music.backend import Backend
from music.frontend import Frontend
from statsmodels.stats.multitest import multipletests
import numpy as np
import yaml


# references for loading the embeddings
with open("fn_suffix.yaml", "r") as stream:
    fn_suffix = yaml.safe_load(stream)

embedding_dimensions = {
    "video": {
        "slow_fast": 2048 if fn_suffix["video"]["slow_fast"] == "_slow" else 256,
    },
    "music": {
        "mfcc": 60,
        "msd": 256,
        "openl3": 512,
    },
    "speech": {
        "hubert": 1024 if fn_suffix["speech"]["hubert"] == "_transformer" else 512,
    },
}

#################
# general utils #
#################


def load_embeddings_and_labels(
    groundtruth_df, emotions_and_mid_level_df, modalities_emb_types, voice, cls_dict
):
    """Load embeddings and labels for given modalities and embedding types.
    Args:
        groundtruth_df (pd.DataFrame): ground truth dataframe containing the classes
        emotions_and_mid_level_df (pd.DataFrame): dataframe containing the regressors
        modalities_emb_types (dict): dictionary containing the modalities and embedding types to load
        voice (bool): whether to load voice or no-voice embeddings
        cls_dict (dict): dictionary containing the SELECTED classes for each classification task
    Returns:
        X (np.array): embeddings
        y_reg (np.array): responses
        y_cls (dict): dictionary containing the labels for each classification task

    Example of cls_dict:
    cls_dict = {
        "target": [
            "Girls/women",
            "Boys/men"
            ], # skip mixed
        "another_column_in_grountruth_df": [
            'selected_class1',
            'selected_class2',
            ...
            ],
        }
    Example of modalities_emb_types:
    modalities_emb_types = {
        "video": ["slow_fast"],
        "music": ["mfcc", "msd", "openl3"],
        "speech": ["hubert"],
    }
    """
    # Load labels
    y_reg = np.empty(
        (emotions_and_mid_level_df.shape[0], emotions_and_mid_level_df.shape[1])
    )
    for i, stimulus_id in enumerate(groundtruth_df.index):
        y_reg[i] = emotions_and_mid_level_df.loc[stimulus_id].values


    X = {}
    for modality in modalities_emb_types.keys():
        X[modality] = {}
        
        for which_emb in modalities_emb_types[modality]:
            X[modality][which_emb] = np.empty((groundtruth_df.shape[0], embedding_dimensions[modality][which_emb]))

            for i, stimulus_id in enumerate(groundtruth_df.index):
                embedding = np.load(
                    f"{modality}/embeddings_{which_emb}{'' if voice else '_novoice'}/"
                    + f"{stimulus_id}{fn_suffix[modality][which_emb]}.npy"
                )
                X[modality][which_emb][i] = embedding.mean(axis=0)


    y_cls = {}
    for task, classes in cls_dict.items():
        y_cls[task] = groundtruth_df[task].values
        y_cls[task] = [classes.index(x) if x in classes else -1 for x in y_cls[task]]
        y_cls[task] = np.array(y_cls[task])

    return X, y_reg, y_cls


def results_to_text(
    all_cls_f1s,
    all_r2s,
    all_pearsons,
    all_ps,
    n_emotions,
    emo_and_mid_cols,
):
    text = ""
    # std across folds, mean across repetitions for each entry in cls_dict
    for k in all_cls_f1s:
        text += f"\t{k} F1: {np.mean(all_cls_f1s[k]):.2f} ± {np.mean(np.std(all_cls_f1s[k], axis=1)):.2f}\n"

    # aggregate and print r2 values, knowing that all_r2s has shape (repetitions, folds, n_regressions)
    text += "\tR2:\n"
    for i, response in enumerate(emo_and_mid_cols):
        text += f"\t\t{response}: {np.mean(all_r2s[:, :, i]):.2f} ± {np.std(all_r2s[:, :, i]):.2f}\n"

    text += "\tPearson's r:\n"
    for i, response in enumerate(emo_and_mid_cols):
        # ratio of significant values with holm-sidak correction
        is_significant = multipletests(
            all_ps[:, :, i].flatten(), alpha=0.05, method="holm-sidak"
        )[0]
        rat_sig = np.sum(is_significant) / len(is_significant)
        temp_txt = f"\t\t{response}: {np.mean(all_pearsons[:, :, i]):.2f} ± {np.std(all_pearsons[:, :, i]):.2f}"
        temp_txt += f" (ratio significant: {rat_sig:.2f})\n"
        text += temp_txt

    # across the emotion responses
    mean_emo_r2 = np.mean(all_r2s[:, :, :n_emotions])
    # std across folds and repetitions, mean across emotions
    std_emo_r2 = np.mean(np.std(all_r2s[:, :, :n_emotions], axis=(0, 1)))

    mean_emo_pears = np.mean(all_pearsons[:, :, :n_emotions])
    std_emo_pears = np.mean(np.std(all_pearsons[:, :, :n_emotions], axis=(0, 1)))

    # across the mid-level responses
    mean_mid_r2 = np.mean(all_r2s[:, :, n_emotions:])
    std_mid_r2 = np.mean(np.std(all_r2s[:, :, n_emotions:], axis=(0, 1)))

    mean_mid_pears = np.mean(all_pearsons[:, :, n_emotions:])
    std_mid_pears = np.mean(np.std(all_pearsons[:, :, n_emotions:], axis=(0, 1)))

    text += f"Average R2 for emotion responses: {mean_emo_r2:.2f} ± {std_emo_r2:.2f}\n"
    text += f"Average Pearson's r for emotion responses: {mean_emo_pears:.2f} ± {std_emo_pears:.2f}\n"
    text += (
        f"Average R2 for mid-level responses: {mean_mid_r2:.2f} ± {std_mid_r2:.2f}\n"
    )
    text += f"Average Pearson's r for mid-level responses: {mean_mid_pears:.2f} ± {std_mid_pears:.2f}\n"

    return text


###################
# MSD model utils #
###################


# compute mel-spectrogram for the MSD model
def compute_melspectrogram(audio_fn, sr=16000, n_fft=512, hop_length=256, n_mels=96):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.core.load(audio_fn, sr=16000, res_type="kaiser_fast")
        spec = librosa.core.amplitude_to_db(
            librosa.feature.melspectrogram(
                y=y,  # type: ignore
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
            )
        )
    return spec


# load pytorch model parameters
def load_parameters(model, filename):
    model = nn.DataParallel(model)  # type: ignore
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
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_reg, y_cls = batch
        y_hat1, y_hat2 = self(x)

        loss1 = nn.MSELoss()(y_hat1, y_reg)

        # Handle missing data in the cost function
        loss2 = F.cross_entropy(y_hat2, y_cls, ignore_index=-1)

        loss = loss1 + loss2
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-5)  # type: ignore


class DynamicDataset(Dataset):
    def __init__(self, X, y_reg, y_cls):
        self.X = {modality: {which_emb: torch.from_numpy(X[modality][which_emb]).float() for which_emb in X[modality]} for modality in X}
        self.y_reg = torch.from_numpy(y_reg).float()
        self.y_cls = {k: torch.from_numpy(v).long() for k, v in y_cls.items()}

    def __len__(self):
        return next(iter(next(iter(self.X.values())).values())).shape[0]

    def __getitem__(self, idx):
        X = {modality: {which_emb: self.X[modality][which_emb][idx] for which_emb in self.X[modality]} for modality in self.X}
        y_cls = {k: v[idx] for k, v in self.y_cls.items()}
        return X, self.y_reg[idx], y_cls


class DynamicMultitasker(LightningModule):
    def __init__(self, routing, n_regressions, cls_dict):
        super().__init__()
        self.routing = routing
        
        self.hidden1_cls = nn.Linear(routing["cls"]["input_dim"], 128)
        self.bn_cls = nn.BatchNorm1d(128)
        self.hidden2_cls = nn.Linear(128, 128)
        self.out_cls = nn.ModuleDict(
            {k: nn.Linear(128, len(v)) for k, v in cls_dict.items()}
        )
        
        self.hidden1_reg = nn.Linear(routing["reg"]["input_dim"], 128)
        self.hidden2_reg = nn.Linear(128, 128)
        self.out_reg = nn.Linear(128, n_regressions)

    def forward(self, x):
        x_cls = F.relu(self.hidden1_cls(x["cls"]))
        x_cls = F.relu(self.hidden2_cls(x_cls))
        x_cls = self.bn_cls(x_cls)
        x_cls = {k: v(x_cls) for k, v in self.out_cls.items()}

        x_reg = F.relu(self.hidden1_reg(x["reg"]))
        x_reg = F.relu(self.hidden2_reg(x_reg))
        x_reg = self.out_reg(x_reg)
        
        return x_reg, x_cls

    def training_step(self, batch, batch_idx):
        x, y_reg, y_cls = batch
        y_hat1, y_hat2 = self(x)

        loss1 = nn.MSELoss()(y_hat1, y_reg)

        # Handle missing data in the cost function
        loss2 = sum(
            [F.cross_entropy(y_hat2[k], v, ignore_index=-1) for k, v in y_cls.items()]
        )

        loss = loss1 + loss2
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_reg, y_cls = batch
        y_hat1, y_hat2 = self(x)

        loss1 = nn.MSELoss()(y_hat1, y_reg)

        # Handle missing data in the cost function
        loss2 = sum(
            [F.cross_entropy(y_hat2[k], v, ignore_index=-1) for k, v in y_cls.items()]
        )

        loss = loss1 + loss2
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=3e-5)
