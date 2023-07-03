import numpy as np
import pandas as pd

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from utils import (
    DynamicDataset,
    DynamicMultitasker,
    load_embeddings_and_labels,
    results_to_dict,
)

from sklearn.model_selection import train_test_split
import yaml, json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config_save.yaml")

args = parser.parse_args()
with open(args.config, "r") as f:
    config = yaml.safe_load(f)


# TODO:
#   - The routing branch is a good idea (different embeddings for different targets).
#     It must not be done by modality, but by voice or no-voice.
#   - a way to include critical analysis in the loss function is to employ a gain on
#     the signal coming from the class gender exaggeration, as well as the scales anger,
#     beauty, etc, together with the overall target.
#   - SEE IF PASSING THE MID-LEVEL FEATURES to the FiLM (not y_hat) works better. 
#     Then at inference you connect the mid-level prediction to the film

torch.manual_seed(42)
#####################
# Load ground truth #
#####################

groundtruth_df = pd.read_csv("groundtruth_merged.csv")
groundtruth_df.set_index("stimulus_id", inplace=True)


emotions_and_mid_level_df = pd.read_csv("emotions_and_mid_level.csv")
emotions_and_mid_level_df.set_index("stimulus_id", inplace=True)

# drop columns that would introduce noise
n_emotions = 7
if config["drop_non_significant"]:
    to_drop = [
        "Amusing",  # Extremely low correlations with all the mid-level features
        "Wide/Narrow pitch variation",  # non significant differences between targets (ANOVA)
        "Repetitive/Non-repetitive",  # non significant differences between targets (ANOVA)
        "Fast tempo/Slow tempo",  # non significant differences between targets (ANOVA)
    ]
    emotions_and_mid_level_df = emotions_and_mid_level_df.drop(columns=to_drop)
    n_emotions -= 1  # we dropped Amusing


###################
# Repeated k-fold #
###################

# iterate over the various configurations

for targets_list in config["targets_list"]:
    for which in config["which_embeddings"]:
        for voice in config["voice_list"]:
            # add current status to config
            config["cls_dict"]["target"] = targets_list

            # load data
            X, y_mid, y_emo, y_cls = load_embeddings_and_labels(
                groundtruth_df,
                emotions_and_mid_level_df,
                which,
                config["modality"],
                voice,
                config["cls_dict"],
                n_emotions,
            )

            # set the parameters for the model
            params = {
                "input_dim": X.shape[1],
                "n_emo": y_emo.shape[1],
                "n_mid": y_mid.shape[1],
                "cls_dict": config["cls_dict"],
                "filmed": False,
            }

            train_index, val_index = train_test_split(
                range(X.shape[0]), test_size=0.15, random_state=42
            )

            # get the split
            X_train, X_val = X[train_index], X[val_index]
            y_mid_train, y_mid_val = y_mid[train_index], y_mid[val_index]
            y_emo_train, y_emo_val = y_emo[train_index], y_emo[val_index]
            y_cls_train, y_cls_val = (
                {k: y_cls[k][train_index] for k in y_cls},
                {k: y_cls[k][val_index] for k in y_cls},
            )
            train_dataset = DynamicDataset(
                X_train, y_mid_train, y_emo_train, y_cls_train
            )
            val_dataset = DynamicDataset(X_val, y_mid_val, y_emo_val, y_cls_val)
            train_loader = DataLoader(
                train_dataset, batch_size=8, shuffle=True, num_workers=1
            )
            val_loader = DataLoader(
                val_dataset, batch_size=8, shuffle=False, num_workers=1
            )

            # train
            model = DynamicMultitasker(**params)

            checkpoint_callback = ModelCheckpoint(monitor="val_loss")
            trainer = pl.Trainer(
                max_epochs=config["max_epochs"],
                callbacks=[
                    checkpoint_callback,
                    EarlyStopping(monitor="val_loss", patience=50),
                ],
                enable_progress_bar=False,
                accelerator="gpu",
                devices=1,
            )
            trainer.fit(model, train_loader, val_loader)

            # load best model
            model = model.load_from_checkpoint(
                checkpoint_callback.best_model_path, **params
            )

            # save model
            torch.save(
                model.state_dict(),
                f"models/{which}_{voice}_voice_{len(targets_list)}_cls.pt",
            )

            # Load model:
            """
            model = DynamicMultitasker(**params)
            model.load_state_dict(torch.load(PATH))
            model.eval()
            """
