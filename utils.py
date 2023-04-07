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
        "openl3_music": 512,
        "openl3_env": 512,
    },
    "speech": {
        "hubert": 1024 if fn_suffix["speech"]["hubert"] == "_transformer" else 512,
    },
}

#################
# general utils #
#################


def load_embeddings_and_labels(
    groundtruth_df,
    emotions_and_mid_level_df,
    which,
    modality,
    voice,
    cls_dict,
    n_emotions=7,
):
    """Load embeddings and labels for a given modality and embedding type.
    Args:
        groundtruth_df (pd.DataFrame): ground truth dataframe containing the classes
        emotions_and_mid_level_df (pd.DataFrame): dataframe containing the regressors
        which (str): which embeddings to load
        modality (str): which modality to load
        voice (bool): whether to load voice or no-voice embeddings
        cls_dict (dict): dictionary containing the SELECTED classes for each classification task
    Returns:
        X (np.array): embeddings
        y_mid (np.array): mid-level features
        y_emo (np.array): emotion scales
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
    """

    # load embeddings
    X = np.empty((groundtruth_df.shape[0], embedding_dimensions[modality][which]))
    y_reg = np.empty(
        (emotions_and_mid_level_df.shape[0], emotions_and_mid_level_df.shape[1])
    )

    for i, stimulus_id in enumerate(groundtruth_df.index):
        embedding = np.load(
            f"{modality}/embeddings_{which}{'' if voice else '_novoice'}/"
            + f"{stimulus_id}{fn_suffix[modality][which]}.npy"
        )
        X[i] = embedding.mean(axis=0)
        y_reg[i] = emotions_and_mid_level_df.loc[stimulus_id].values

    y_mid, y_emo = y_reg[:, n_emotions:], y_reg[:, :n_emotions]

    y_cls = {}
    for task, classes in cls_dict.items():
        y_cls[task] = groundtruth_df[task].values
        y_cls[task] = [classes.index(x) if x in classes else -1 for x in y_cls[task]]
        y_cls[task] = np.array(y_cls[task])

    return X, y_mid, y_emo, y_cls


def results_to_text(
    all_cls_f1s,
    all_r2s_mid,
    all_r2s_emo,
    all_pear_mid,
    all_pear_emo,
    all_ps_mid,
    all_ps_emo,
    mid_cols,
    emo_cols,
):
    text = ""
    # std across folds, mean across repetitions for each entry in cls_dict
    for k in all_cls_f1s:
        text += f"\t{k} F1: {np.mean(all_cls_f1s[k]):.2f} ± {np.mean(np.std(all_cls_f1s[k], axis=1)):.2f}\n"

    # aggregate and print r2 values, knowing that all_r2s has shape (repetitions, folds, n_regressions)
    text += "\tR2:\n"
    for i, response in enumerate(mid_cols):
        text += f"\t\t{response}: {np.mean(all_r2s_mid[:, :, i]):.2f} ± {np.std(all_r2s_mid[:, :, i]):.2f}\n"

    for i, response in enumerate(emo_cols):
        text += f"\t\t{response}: {np.mean(all_r2s_emo[:, :, i]):.2f} ± {np.std(all_r2s_emo[:, :, i]):.2f}\n"

    text += "\tPearson's r:\n"
    for i, response in enumerate(mid_cols):
        # ratio of significant values with holm-sidak correction
        is_significant = multipletests(
            all_ps_mid[:, :, i].flatten(), alpha=0.05, method="holm-sidak"
        )[0]
        rat_sig = np.sum(is_significant) / len(is_significant)
        temp_txt = f"\t\t{response}: {np.mean(all_pear_mid[:, :, i]):.2f} ± {np.std(all_pear_mid[:, :, i]):.2f}"
        temp_txt += f" (ratio significant: {rat_sig:.2f})\n"
        text += temp_txt

    for i, response in enumerate(emo_cols):
        # ratio of significant values with holm-sidak correction
        is_significant = multipletests(
            all_ps_emo[:, :, i].flatten(), alpha=0.05, method="holm-sidak"
        )[0]
        rat_sig = np.sum(is_significant) / len(is_significant)
        temp_txt = f"\t\t{response}: {np.mean(all_pear_emo[:, :, i]):.2f} ± {np.std(all_pear_emo[:, :, i]):.2f}"
        temp_txt += f" (ratio significant: {rat_sig:.2f})\n"
        text += temp_txt

    # across the mid-level responses
    mean_mid_r2 = np.mean(all_r2s_mid)
    std_mid_r2 = np.mean(np.std(all_r2s_mid, axis=(0, 1)))

    mean_mid_pears = np.mean(all_pear_mid)
    std_mid_pears = np.mean(np.std(all_pear_mid, axis=(0, 1)))

    # across the emotion responses
    mean_emo_r2 = np.mean(all_r2s_emo)
    # std across folds and repetitions, mean across emotions
    std_emo_r2 = np.mean(np.std(all_r2s_emo, axis=(0, 1)))

    mean_emo_pears = np.mean(all_pear_emo)
    std_emo_pears = np.mean(np.std(all_pear_emo, axis=(0, 1)))

    text += (
        f"Average R2 for mid-level responses: {mean_mid_r2:.2f} ± {std_mid_r2:.2f}\n"
    )
    text += f"Average Pearson's r for mid-level responses: {mean_mid_pears:.2f} ± {std_mid_pears:.2f}\n"
    text += f"Average R2 for emotion responses: {mean_emo_r2:.2f} ± {std_emo_r2:.2f}\n"
    text += f"Average Pearson's r for emotion responses: {mean_emo_pears:.2f} ± {std_emo_pears:.2f}\n"

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


class FiLM(LightningModule):
    def __init__(self, num_features, num_outputs):
        super(FiLM, self).__init__()
        self.fc_gamma = nn.Linear(num_features, num_outputs)
        self.fc_beta = nn.Linear(num_features, num_outputs)

    def forward(self, x, cond):
        gamma = self.fc_gamma(cond)
        beta = self.fc_beta(cond)
        return x * gamma + beta


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
    def __init__(self, X, y_mid, y_emo, y_cls):
        self.X = torch.from_numpy(X).float()
        self.y_mid = torch.from_numpy(y_mid).float()
        self.y_emo = torch.from_numpy(y_emo).float()
        self.y_cls = {k: torch.from_numpy(v).long() for k, v in y_cls.items()}

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        y_cls = {k: v[idx] for k, v in self.y_cls.items()}
        return self.X[idx], self.y_mid[idx], self.y_emo[idx], y_cls


class DynamicMultitasker(LightningModule):
    def __init__(self, input_dim, n_mid, n_emo, cls_dict, cls_weighing=1.0):
        super().__init__()
        self.cls_weighing = cls_weighing
        self.hidden = nn.Linear(input_dim, 128)

        self.hidden_mid = nn.Linear(128, 128)
        self.out_mid = nn.Linear(128, n_mid)
        self.film = FiLM(n_mid, 128)

        self.hidden_emo = nn.Linear(128, 128)
        self.out_emo = nn.Linear(128, n_emo)

        self.bn_cls = nn.BatchNorm1d(128)
        self.hidden_cls = nn.Linear(128, 128)
        self.out_cls = nn.ModuleDict(
            {k: nn.Linear(128, len(v)) for k, v in cls_dict.items()}
        )

    def forward(self, x):
        x = F.relu(self.hidden(x))

        x_cls = self.bn_cls(x)
        x_cls = F.relu(self.hidden_cls(x))
        x_cls = {k: v(x_cls) for k, v in self.out_cls.items()}

        x_mid = F.relu(self.hidden_mid(x))
        x_mid = self.out_mid(x_mid)

        x_emo = F.relu(self.hidden_emo(x))
        # condition the output of hidden_emo on the mid-level features
        x_emo = self.film(x_emo, x_mid)
        x_emo = self.out_emo(x_emo)

        return x_mid, x_emo, x_cls

    def training_step(self, batch, batch_idx):
        x, y_mid, y_emo, y_cls = batch
        y_hat1, y_hat2, y_hat3 = self(x)

        loss_mid = nn.MSELoss()(y_hat1, y_mid)
        loss_emo = nn.MSELoss()(y_hat2, y_emo)

        # Handle missing data in the cost function
        loss_cls = sum(
            [F.cross_entropy(y_hat3[k], v, ignore_index=-1) for k, v in y_cls.items()]
        )

        loss = loss_mid + loss_emo + self.cls_weighing * loss_cls
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_mid, y_emo, y_cls = batch
        y_hat1, y_hat2, y_hat3 = self(x)

        loss_mid = nn.MSELoss()(y_hat1, y_mid)
        loss_emo = nn.MSELoss()(y_hat2, y_emo)

        # Handle missing data in the cost function
        loss_cls = sum(
            [F.cross_entropy(y_hat3[k], v, ignore_index=-1) for k, v in y_cls.items()]
        )

        loss = loss_mid + loss_emo + self.cls_weighing * loss_cls
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-3, amsgrad=True)  # type: ignore
