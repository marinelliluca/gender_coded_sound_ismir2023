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
    embedding_dimensions,
    results_to_text,
)

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, f1_score
from scipy.stats import pearsonr
import yaml

# set torch seed
torch.manual_seed(42)

# TODO:
#   - The routing branch is a good idea (different embeddings for different targets).
#     It must not be done by modality, but by voice or no-voice.
#   - a way to do include critical analysis in the loss function is to employ a gain on
#     the signal coming from the class gender exaggeration, as well as the scales anger,
#     beauty, etc, together with the overall target.

# load config
with open("config.yaml", "r") as stream:
    config = yaml.safe_load(stream)

config["targets_list"] = [
    ["Girls/women", "Boys/men"],
    ["Girls/women", "Mixed", "Boys/men"],
]

if config["target_behaviour"] == "iterate":
    pass
elif config["target_behaviour"] == "binary":
    _ = config["targets_list"].pop(1)
elif config["target_behaviour"] == "ternary":
    _ = config["targets_list"].pop(0)
else:
    raise ValueError("target_behaviour not valid")

#####################
# Load ground truth #
#####################

groundtruth_df = pd.read_csv("groundtruth_merged.csv")
groundtruth_df.set_index("stimulus_id", inplace=True)

# load responses
emotions_and_mid_level_df = pd.read_csv("emotions_and_mid_level.csv")
emotions_and_mid_level_df.set_index("stimulus_id", inplace=True)

n_emotions = 7
if config["drop_non_significant"]:
    # drop columns that are not significant based on the ANOVA test
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
            config["cls_dict"]["target"] = targets_list

            X, y_mid, y_emo, y_cls = load_embeddings_and_labels(
                groundtruth_df,
                emotions_and_mid_level_df,
                which,
                config["modality"],
                voice,
                config["cls_dict"],
                n_emotions,
            )

            params = {
                "input_dim": X.shape[1],
                "n_emo": y_emo.shape[1],
                "n_mid": y_mid.shape[1],
                "cls_dict": config["cls_dict"],
            }

            # all_accuracies = []
            all_cls_f1s = {k: [] for k in config["cls_dict"]}

            all_r2s_mid = []
            all_r2s_emo = []
            all_pear_mid = []
            all_pear_emo = []
            all_ps_mid = []
            all_ps_emo = []

            for _ in range(config["repetitions"]):
                kf = KFold(
                    n_splits=config["folds"], shuffle=True
                )  # NO RANDOM SEED, get a new split each time

                cls_f1s = {k: [] for k in config["cls_dict"]}
                r2s_mid = []
                r2s_emo = []
                pears_mid = []
                pears_emo = []
                ps_mid = []
                ps_emo = []

                for train_index, test_index in kf.split(X):
                    test_index, val_index = train_test_split(test_index, test_size=0.5)

                    X_train, X_test, X_val = X[train_index], X[test_index], X[val_index]

                    y_mid_train, y_mid_test, y_mid_val = (
                        y_mid[train_index],
                        y_mid[test_index],
                        y_mid[val_index],
                    )

                    y_emo_train, y_emo_test, y_emo_val = (
                        y_emo[train_index],
                        y_emo[test_index],
                        y_emo[val_index],
                    )

                    y_cls_train, y_cls_test, y_cls_val = (
                        {k: y_cls[k][train_index] for k in y_cls},
                        {k: y_cls[k][test_index] for k in y_cls},
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

                    model = DynamicMultitasker(**params)

                    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
                    trainer = pl.Trainer(
                        max_epochs=config["max_epochs"],
                        callbacks=[
                            checkpoint_callback,
                            EarlyStopping(monitor="val_loss", patience=30),
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

                    # evaluate on test set
                    model.eval()
                    with torch.no_grad():
                        y_mid_pred, y_emo_pred, y_cls_pred = model(
                            torch.from_numpy(X_test).float()
                        )

                    for k in config["cls_dict"]:
                        y_pred_temp = y_cls_pred[k]
                        y_test_temp = y_cls_test[k]
                        skip_unlabelled = y_test_temp != -1
                        y_pred_temp = torch.argmax(y_pred_temp, dim=1).numpy()[
                            skip_unlabelled
                        ]
                        y_test_temp = y_test_temp[skip_unlabelled]
                        cls_f1s[k].append(
                            f1_score(y_test_temp, y_pred_temp, average="weighted")
                        )

                    y_mid_pred = y_mid_pred.numpy()
                    y_emo_pred = y_emo_pred.numpy()
                    r2_mid = r2_score(y_mid_test, y_mid_pred, multioutput="raw_values")
                    r2s_mid.append(r2_mid)
                    r2_emo = r2_score(y_emo_test, y_emo_pred, multioutput="raw_values")
                    r2s_emo.append(r2_emo)

                    r_mid = [
                        pearsonr(y_mid_test[:, i], y_mid_pred[:, i])[0]
                        for i in range(y_mid_test.shape[1])
                    ]

                    r_emo = [
                        pearsonr(y_emo_test[:, i], y_emo_pred[:, i])[0]
                        for i in range(y_emo_test.shape[1])
                    ]

                    p_mid = [
                        pearsonr(y_mid_test[:, i], y_mid_pred[:, i])[1]
                        for i in range(y_mid_test.shape[1])
                    ]

                    p_emo = [
                        pearsonr(y_emo_test[:, i], y_emo_pred[:, i])[1]
                        for i in range(y_emo_test.shape[1])
                    ]

                    pears_emo.append(r_emo)
                    pears_mid.append(r_mid)
                    ps_emo.append(p_emo)
                    ps_mid.append(p_mid)

                for k in config["cls_dict"]:
                    all_cls_f1s[k].append(cls_f1s[k])

                all_r2s_emo.append(r2s_emo)
                all_r2s_mid.append(r2s_mid)
                all_pear_emo.append(pears_emo)
                all_pear_mid.append(pears_mid)
                all_ps_emo.append(ps_emo)
                all_ps_mid.append(ps_mid)

            # convert to numpy arrays
            for k in config["cls_dict"]:
                all_cls_f1s[k] = np.array(all_cls_f1s[k])

            all_r2s_mid = np.array(all_r2s_mid)
            all_r2s_emo = np.array(all_r2s_emo)
            all_pear_mid = np.array(all_pear_mid)
            all_pear_emo = np.array(all_pear_emo)
            all_ps_mid = np.array(all_ps_mid)
            all_ps_emo = np.array(all_ps_emo)

            #################
            # Print results #
            #################

            sec_classfc = [k for k in config["cls_dict"] if k != "target"]
            text = f"Target + regressions + {', '.join(sec_classfc)}\n"
            text += f"\tTarget classes: {len(targets_list)}\n"
            text += f"\tVoice: {voice}\n"
            text += (
                f"Drop non significant regressors: {config['drop_non_significant']}\n"
            )
            text += f"Embeddings: {which}, (modality: {config['modality']})\n"

            text += results_to_text(
                all_cls_f1s,
                all_r2s_mid,
                all_r2s_emo,
                all_pear_mid,
                all_pear_emo,
                all_ps_mid,
                all_ps_emo,
                emotions_and_mid_level_df.columns[n_emotions:],
                emotions_and_mid_level_df.columns[:n_emotions]
            )

            print(text)

            # save results to file

            # identifiable filename with all relevant parameters
            filename = f"results_filmed/targCls_{len(targets_list)}_{config['modality']}_{which}_voice_{voice}_NsecCls_{len(sec_classfc)}_"
            filename += f"dropNs_{config['drop_non_significant']}_rep_{config['repetitions']}_fold_{config['folds']}.txt"
            with open(filename, "a") as f:
                f.write(text)
