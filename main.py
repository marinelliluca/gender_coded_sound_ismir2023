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

""" I want the code to be able to handle the output of the final and complete fom of load_embeddings_and_labels, 
so that x_cls and x_reg can be assigned to different embeddings, according to a `routing` dictionary that will be stored 
in self.routing within DynamicMultitasker. 
Example of usage: 
routing = { 
    "cls": { "X": X[modality_cls][emb_cls], "input_dim": X[modality_cls][emb_cls].shape[0]}, 
    "reg": { "X": X[modality_reg][em_reg], "input_dim": X[modality_reg][em_reg].shape[0]} 
} 
model = DynamicMultitasker(routing, n_regressions, cls_dict) """


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

            X, y_reg, y_cls = load_embeddings_and_labels(
                groundtruth_df,
                emotions_and_mid_level_df,
                which,
                config["modality"],
                voice,
                config["cls_dict"],
            )

            params = {
                "input_dim": X.shape[1],
                "n_regressions": y_reg.shape[1],
                "cls_dict": config["cls_dict"],
            }

            # all_accuracies = []
            all_cls_f1s = {k: [] for k in config["cls_dict"]}

            all_r2s = []
            all_pearsons = []
            all_ps = []

            for _ in range(config["repetitions"]):
                kf = KFold(
                    n_splits=config["folds"], shuffle=True
                )  # get a new split each time

                cls_f1s = {k: [] for k in config["cls_dict"]}
                r2s = []
                pearsons = []
                ps = []

                for train_index, test_index in kf.split(X):
                    test_index, val_index = train_test_split(test_index, test_size=0.5)

                    X_train, X_test, X_val = X[train_index], X[test_index], X[val_index]
                    y_reg_train, y_reg_test, y_reg_val = (
                        y_reg[train_index],
                        y_reg[test_index],
                        y_reg[val_index],
                    )

                    y_cls_train, y_cls_test, y_cls_val = (
                        {k: y_cls[k][train_index] for k in y_cls},
                        {k: y_cls[k][test_index] for k in y_cls},
                        {k: y_cls[k][val_index] for k in y_cls},
                    )

                    train_dataset = DynamicDataset(X_train, y_reg_train, y_cls_train)
                    val_dataset = DynamicDataset(X_val, y_reg_val, y_cls_val)

                    train_loader = DataLoader(
                        train_dataset, batch_size=8, shuffle=True, num_workers=1
                    )
                    val_loader = DataLoader(
                        val_dataset, batch_size=8, shuffle=False, num_workers=1
                    )

                    model = DynamicMultitasker(**params)

                    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
                    trainer = pl.Trainer(
                        max_epochs=100,
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
                        y_hat1, y_hat2 = model(torch.from_numpy(X_test).float())

                    for k in config["cls_dict"]:
                        y_pred_temp = y_hat2[k]
                        y_test_temp = y_cls_test[k]
                        skip_unlabelled = y_test_temp != -1
                        y_pred_temp = torch.argmax(y_pred_temp, dim=1).numpy()[
                            skip_unlabelled
                        ]
                        y_test_temp = y_test_temp[skip_unlabelled]
                        cls_f1s[k].append(
                            f1_score(y_test_temp, y_pred_temp, average="weighted")
                        )

                    y_reg_pred = y_hat1.numpy()
                    r2_values = r2_score(
                        y_reg_test, y_reg_pred, multioutput="raw_values"
                    )
                    r2s.append(r2_values)

                    r = [
                        pearsonr(y_reg_test[:, i], y_reg_pred[:, i])[0]
                        for i in range(y_reg_test.shape[1])
                    ]
                    p = [
                        pearsonr(y_reg_test[:, i], y_reg_pred[:, i])[1]
                        for i in range(y_reg_test.shape[1])
                    ]
                    pearsons.append(r)
                    ps.append(p)

                for k in config["cls_dict"]:
                    all_cls_f1s[k].append(cls_f1s[k])

                all_r2s.append(r2s)
                all_pearsons.append(pearsons)
                all_ps.append(ps)

            # convert to numpy arrays
            for k in config["cls_dict"]:
                all_cls_f1s[k] = np.array(all_cls_f1s[k])

            all_r2s = np.array(all_r2s)
            all_pearsons = np.array(all_pearsons)
            all_ps = np.array(all_ps)

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
                all_r2s,
                all_pearsons,
                all_ps,
                n_emotions,
                emotions_and_mid_level_df.columns,
            )

            print(text)

            # save results to file

            # identifiable filename with all relevant parameters
            filename = f"results/targCls_{len(targets_list)}_{config['modality']}_{which}_voice_{voice}_NsecCls_{len(sec_classfc)}_"
            filename += f"dropNs_{config['drop_non_significant']}_rep_{config['repetitions']}_fold_{config['folds']}.txt"
            with open(filename, "a") as f:
                f.write(text)
