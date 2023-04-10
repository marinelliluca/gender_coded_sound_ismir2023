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
from tabulate import tabulate
import pandas as pd
import glob
import json

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
        emotions_and_mid_level_df (pd.DataFrame): dataframe containing the responses

            ===>> NB: emotions first, then mid-level features

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



def results_to_dict(
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
    result = {}
    
    # std across folds, mean across repetitions for each entry in cls_dict
    result['f1'] = {}
    for k in all_cls_f1s:
        result['f1'][k] = {
            'mean': np.mean(all_cls_f1s[k]),
            'std': np.mean(np.std(all_cls_f1s[k], axis=1))
        }

    # aggregate and print r2 values, knowing that all_r2s has shape (repetitions, folds, n_regressions)
    result['r2'] = {}
    for i, response in enumerate(mid_cols + emo_cols):
        if i < len(mid_cols):
            result['r2'][response] = {
                'mean': np.mean(all_r2s_mid[:, :, i]),
                'std': np.std(all_r2s_mid[:, :, i])
            }
        else:
            result['r2'][response] = {
                'mean': np.mean(all_r2s_emo[:, :, i - len(mid_cols)]),
                'std': np.std(all_r2s_emo[:, :, i - len(mid_cols)])
            }

    result['pearson'] = {}
    for i, response in enumerate(mid_cols + emo_cols):

        if i < len(mid_cols):
            # ratio of significant values with holm-sidak correction
            is_significant = multipletests(
                all_ps_mid[:, :, i].flatten(), alpha=0.05, method="holm-sidak"
            )[0]
            rat_sig = np.sum(is_significant) / len(is_significant)

            result['pearson'][response] = {
                'mean': np.mean(all_pear_mid[:, :, i]),
                'std': np.std(all_pear_mid[:, :, i]),
                'ratio_significant': rat_sig
            }
        else:
            is_significant = multipletests(
                all_ps_emo[:, :, i - len(mid_cols)].flatten(), alpha=0.05, method="holm-sidak"
            )[0]
            rat_sig = np.sum(is_significant) / len(is_significant)

            result['pearson'][response] = {
                'mean': np.mean(all_pear_emo[:, :, i - len(mid_cols)]),
                'std': np.std(all_pear_emo[:, :, i - len(mid_cols)]),
                'ratio_significant': rat_sig
            }
    
    result['average'] = {
        'mid_r2': {
            # mean across folds and repetitions
            'mean': np.mean(all_r2s_mid),
            # std across folds and repetitions, mean across mid-level responses
            'std': np.mean(np.std(all_r2s_mid, axis=(0, 1)))
        },
        'mid_pears': {
            'mean': np.mean(all_pear_mid),
            'std': np.mean(np.std(all_pear_mid, axis=(0, 1)))
        },
        'emo_r2': {
            'mean': np.mean(all_r2s_emo),
            'std': np.mean(np.std(all_r2s_emo, axis=(0, 1)))
        },
        'emo_pears': {
            'mean': np.mean(all_pear_emo),
            'std': np.mean(np.std(all_pear_emo, axis=(0, 1)))
        }
    }

    return result

def display_results(dictionary):
    rows = []
    config_rows = []
    f1_rows = []
    r2_rows = []
    pearson_rows = []
    average_rows = []
    for key, value in dictionary.items():
        if key == 'config':
            target_rows = config_rows
        elif key == 'f1':
            target_rows = f1_rows
            
            # aggregate secondary f1 scores
            f1_secondary_means = [v['mean'] for k,v in value.items() if k != 'target']
            f1_secondary_stds = [v['std'] for k,v in value.items() if k != 'target']
            
        elif key == 'r2':
            target_rows = r2_rows
        elif key == 'pearson':
            target_rows = pearson_rows
        elif key == 'average':
            target_rows = average_rows
        else:
            target_rows = rows
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                if isinstance(subvalue, dict):
                    for subsubkey, subsubvalue in subvalue.items():
                        target_rows.append([key, subkey, subsubkey, format_value(subsubvalue)])
                        key = ''
                        subkey = ''
                else:
                    target_rows.append([key, subkey, '', format_value(subvalue)])
                    key = ''
        else:
            target_rows.append([key, '', '', format_value(value)])
    
    # add f1 secondary scores
    average_rows.append(['', 'secondary f1', 'mean', format_value(np.mean(f1_secondary_means))])
    average_rows.append(['', '', 'std', format_value(np.mean(f1_secondary_stds))])

    final_rows = config_rows + average_rows + f1_rows + r2_rows + pearson_rows + rows
    print(tabulate(final_rows,headers=['Key', 'Subkey', 'Sub-subkey', 'Value']))
    #return average_rows

def format_value(value):
    if isinstance(value,(int,float)):
        return f'{value:.2f}'
    elif isinstance(value,list):
        return f'{len(value)} cls'
    else:
        return value

def display_structure(dictionary):
    rows = []
    for key, value in dictionary.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                if isinstance(subvalue, dict):
                    for subsubkey, subsubvalue in subvalue.items():
                        rows.append([key, subkey, subsubkey, type(subsubvalue).__name__])
                        key = ''
                        subkey = ''
                else:
                    rows.append([key, subkey, '', type(subvalue).__name__])
                    key = ''
                    subkey = ''
        else:
            rows.append([key, '', '', type(value).__name__])
    print(tabulate(rows, headers=['Key', 'Subkey', 'Sub-subkey', 'Type']))

def experiments_to_dict_of_dfs(folder, rep, fold): 
    target_task = ["Binary", "Ternary"]
    voice_task = ["yes", "no"]
    metrics = ["Target F1", "Avg. secondary F1", "Avg. R2 emotions", "Avg. r emotions", "Avg. R2 mid-level", "Avg. r mid-level"]
    columns = ["Embeddings", "Voice"] + metrics

    dict_of_dfs = {
        tt: pd.DataFrame(columns=columns)
        for tt in target_task
    }

    for fn in glob.glob(f"./{folder}/*_rep_{rep}_fold_{fold}.json"):
        with open(fn) as f:
            results = json.load(f)

        # Compute average secondary F1 score
        f1_secondary_means = [v['mean'] for k,v in results["f1"].items() if k != 'target']
        f1_secondary_stds = [v['std'] for k,v in results["f1"].items() if k != 'target']
        results["average"]["secondary f1"] = { "mean": np.mean(f1_secondary_means), "std": np.mean(f1_secondary_stds)}

        # Extract relevant metrics
        row = {
            "Embeddings": results["config"]["which_embeddings"],
            "Voice": "yes" if results["config"]["voice"] else "no",
            "Target F1": f'{results["f1"]["target"]["mean"]:.2f} ± {results["f1"]["target"]["std"]:.2f}',
            "Avg. secondary F1": f'{results["average"]["secondary f1"]["mean"]:.2f} ± {results["average"]["secondary f1"]["std"]:.2f}',
            "Avg. R2 emotions": f'{results["average"]["emo_r2"]["mean"]:.2f} ± {results["average"]["emo_r2"]["std"]:.2f}',
            "Avg. r emotions": f'{results["average"]["emo_pears"]["mean"]:.2f} ± {results["average"]["emo_pears"]["std"]:.2f}',
            "Avg. R2 mid-level": f'{results["average"]["mid_r2"]["mean"]:.2f} ± {results["average"]["mid_r2"]["std"]:.2f}',
            "Avg. r mid-level": f'{results["average"]["mid_pears"]["mean"]:.2f} ± {results["average"]["mid_pears"]["std"]:.2f}'
        }

        # reduce the width of the table by taking out the 0.
        row = {k: v.replace("0.", ".") for k, v in row.items()}

        # Determine indices for dict_of_dfs
        tt = "Binary" if len(results["config"]["classifications"]["target"]) == 2 else "Ternary"

        # Add row to the appropriate DataFrame
        df = dict_of_dfs[tt].copy()
        dict_of_dfs[tt] = pd.concat([df, pd.DataFrame(row, index=[0])], ignore_index=True)

    for tt in dict_of_dfs:
        df = dict_of_dfs[tt].copy()
        dict_of_dfs[tt] = df.pivot(index="Embeddings", columns="Voice")

    return dict_of_dfs

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
        # the idea was to have the following:
        # y_emo = y_mid * gamma(x) + beta(x)
        return x * gamma + beta

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
    def __init__(self, input_dim, n_mid, n_emo, cls_dict, filmed=True, cls_weighing=1.0):
        super().__init__()

        # TODO: 
        # - in future studies we should try see how cls_weighing affects the results
        # - also see whether a granular weighting for each output would be better

        self.cls_weighing = cls_weighing
        self.filmed = filmed
        
        self.hidden = nn.Linear(input_dim, 128)

        self.hidden_mid = nn.Linear(128, 128)
        self.out_mid = nn.Linear(128, n_mid)

        if self.filmed:
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
        if self.filmed:
            x_emo = self.film(x_emo, x_mid)
        x_emo = self.out_emo(x_emo)

        return x_mid, x_emo, x_cls

    def training_step(self, batch, batch_idx):
        x, y_mid, y_emo, y_cls = batch
        y_hat_mid, y_hat_emo, y_hat_cls = self(x)

        loss_mid = nn.MSELoss()(y_hat_mid, y_mid)
        loss_emo = nn.MSELoss()(y_hat_emo, y_emo)

        # Handle missing data in the cost function
        loss_cls = sum(
            [F.cross_entropy(y_hat_cls[k], v, ignore_index=-1) for k, v in y_cls.items()]
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
