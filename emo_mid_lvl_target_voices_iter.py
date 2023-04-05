import numpy as np
import pandas as pd

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from utils import (
    MegaModelV1,
    EmbeddingsDataset2,
    load_embeddings_and_labels,
)

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
import yaml

""" # load config
with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

# assign a variable to each key in the config dictionary for convenience
n_tar_cls = config["n_tar_cls"]
modality = config["modality"]
which = config["which"]
drop_non_significant = config["drop_non_significant"]
voice = config["voice"]
repetitions = config["repetitions"]
folds = config["folds"] """

modality = "music"
drop_non_significant = True
repetitions = 10
folds = 5

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

#####################
# Load ground truth #
#####################

groundtruth_df = pd.read_csv("groundtruth_merged.csv")
groundtruth_df.set_index("stimulus_id", inplace=True)

# load responses
emotions_and_mid_level = pd.read_csv("emotions_and_mid_level.csv")
emotions_and_mid_level.set_index("stimulus_id", inplace=True)

n_emotions = 7

if drop_non_significant:
    # drop columns that are not significant based on the ANOVA test
    to_drop = [
        "Amusing",  # Extremely low correlations with all the mid-level features
        "Wide/Narrow pitch variation",  # non significant differences between targets (ANOVA)
        "Repetitive/Non-repetitive",  # non significant differences between targets (ANOVA)
        "Fast tempo/Slow tempo",  # non significant differences between targets (ANOVA)
    ]
    emotions_and_mid_level = emotions_and_mid_level.drop(columns=to_drop)
    n_emotions -= 1  # we dropped Amusing


# iterate over the various configurations
for n_tar_cls in [2,3]:
    for which in ["mfcc", "msd", "openl3"]:
        ##############################
        # Load embeddings and labels #
        ##############################

        X, y_reg, y_tar, y_voig = load_embeddings_and_labels(
            groundtruth_df,
            emotions_and_mid_level,
            embedding_dimensions[modality][which],
            which,
            modality,
            True,
            n_tar_cls,
        )

        #############
        # K-Fold CV #
        #############

        params = {
            "input_dim": X.shape[1],
            "n_regressions": y_reg.shape[1],
            "output_dim1": len(np.unique(y_tar)),
            "output_dim2": len(np.unique(y_voig)),
        }

        # all_accuracies = []
        all_tar_f1s = []
        all_voig_f1s = []
        all_r2s = []
        all_pearsons = []
        all_ps = []

        for _ in range(repetitions):
            kf = KFold(n_splits=folds, shuffle=True)  # get a new split each time

            # accuracies = []
            tar_f1s = []
            voig_f1s = []
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
                y_tar_train, y_tar_test, y_tar_val = (
                    y_tar[train_index],
                    y_tar[test_index],
                    y_tar[val_index],
                )
                y_voig_train, y_voig_test, y_voig_val = (
                    y_voig[train_index],
                    y_voig[test_index],
                    y_voig[val_index],
                )

                train_dataset = EmbeddingsDataset2(X_train, y_reg_train, y_tar_train,y_voig_train )
                val_dataset = EmbeddingsDataset2(X_val,y_reg_val ,y_tar_val ,y_voig_val )

                train_loader = DataLoader(
                    train_dataset, batch_size=8, shuffle=True, num_workers=1
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=8, shuffle=False, num_workers=1
                )

                model = MegaModelV1(**params)

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
                    y_reg_pred, y_tar_pred, y_voig_pred = model(torch.from_numpy(X_test).float())

                skip_unlabelled = y_tar_test != -1
                y_tar_pred = torch.argmax(y_tar_pred, dim=1).numpy()[skip_unlabelled]
                y_tar_test = y_tar_test[skip_unlabelled]
                tar_f1s.append(f1_score(y_tar_test, y_tar_pred, average="weighted"))
                

                skip_unlabelled = y_voig_test != -1
                y_voig_pred = torch.argmax(y_voig_pred, dim=1).numpy()[skip_unlabelled]
                y_voig_test = y_voig_test[skip_unlabelled]
                voig_f1s.append(f1_score(y_voig_test, y_voig_pred, average="weighted"))

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

            # all_accuracies.append(accuracies)
            all_tar_f1s.append(tar_f1s)
            all_voig_f1s.append(voig_f1s)
            all_r2s.append(r2s)
            all_pearsons.append(pearsons)
            all_ps.append(ps)

        # all_accuracies = np.array(all_accuracies)
        all_tar_f1s = np.array(all_tar_f1s)
        all_voig_f1s = np.array(all_voig_f1s)
        all_r2s = np.array(all_r2s)
        all_pearsons = np.array(all_pearsons)
        all_ps = np.array(all_ps)

        #################
        # Print results #
        #################

        print(
            f"n_tar_cls: {n_tar_cls}, modality: {modality}, which: {which}, repetitions: {repetitions}"
        )

        # std across folds, mean across repetitions
        print(
            f"\tTarget F1: {np.mean(all_tar_f1s):.2f} ± {np.mean(np.std(all_tar_f1s, axis=1)):.2f}"
        )

        print(
            f"\tVoice Gender F1: {np.mean(all_voig_f1s):.2f} ± {np.mean(np.std(all_voig_f1s, axis=1)):.2f}"
        )

        # aggregate and print r2 values, knowing that all_r2s has shape (repetitions, folds, n_regressions)
        print("\tR2:")
        for i, response in enumerate(emotions_and_mid_level.columns):
            print(
                f"\t\t{response}: {np.mean(all_r2s[:, :, i]):.2f} ± {np.std(all_r2s[:, :, i]):.2f}"
            )

        print("\tPearson's r:")
        for i, response in enumerate(emotions_and_mid_level.columns):
            # ratio of significant values with holm-sidak correction
            is_significant = multipletests(
                all_ps[:, :, i].flatten(), alpha=0.05, method="holm-sidak"
            )[0]
            rat_sig = np.sum(is_significant) / len(is_significant)

            print(
                f"\t\t{response}: {np.mean(all_pearsons[:, :, i]):.2f} ± {np.std(all_pearsons[:, :, i]):.2f} (ratio significant: {rat_sig:.2f})"
            )

        # across the emotion responses
        mean_emo_r2 = np.mean(all_r2s[:, :, :n_emotions])
        # std across folds and repetitions, mean across emotions
        std_emo_r2 = np.mean(np.std(all_r2s[:, :, :n_emotions], axis=(0, 1)))

        mean_emo_pears = np.mean(all_pearsons[:, :, :n_emotions])
        std_emo_pears = np.mean(
            np.std(all_pearsons[:, :, :n_emotions], axis=(0, 1))
        )

        # across the mid-level responses
        mean_mid_r2 = np.mean(all_r2s[:, :, n_emotions:])
        std_mid_r2 = np.mean(np.std(all_r2s[:, :, n_emotions:], axis=(0, 1)))

        mean_mid_pears = np.mean(all_pearsons[:, :, n_emotions:])
        std_mid_pears = np.mean(
            np.std(all_pearsons[:, :, n_emotions:], axis=(0, 1))
        )

        print(
            f"Average R2 for emotion responses: {mean_emo_r2:.2f} ± {std_emo_r2:.2f}"
        )
        print(
            f"Average Pearson's r for emotion responses: {mean_emo_pears:.2f} ± {std_emo_pears:.2f}"
        )

        print(
            f"Average R2 for mid-level responses: {mean_mid_r2:.2f} ± {std_mid_r2:.2f}"
        )
        print(
            f"Average Pearson's r for mid-level responses: {mean_mid_pears:.2f} ± {std_mid_pears:.2f}"
        )

        # now write the results to a file
        filename = f"voige_gender_results_{n_tar_cls}_{modality}_{which}.txt"
        with open(filename, "a") as f:
            f.write(
                f"n_tar_cls: {n_tar_cls}, modality: {modality}, which: {which}, repetitions: {repetitions}\n"
            )
            f.write(
                f"\tTarget F1: {np.mean(all_tar_f1s):.2f} ± {np.mean(np.std(all_tar_f1s, axis=1)):.2f}\n"
            )
            f.write(
                f"\tVoice Gender F1: {np.mean(all_voig_f1s):.2f} ± {np.mean(np.std(all_voig_f1s, axis=1)):.2f}"
            )    
            f.write("\tR2:\n")
            for i, response in enumerate(emotions_and_mid_level.columns):
                f.write(
                    f"\t\t{response}: {np.mean(all_r2s[:, :, i]):.2f} ± {np.std(all_r2s[:, :, i]):.2f}\n"
                )
            f.write("\tPearson's r:\n")
            for i, response in enumerate(emotions_and_mid_level.columns):
                f.write(
                    f"\t\t{response}: {np.mean(all_pearsons[:, :, i]):.2f} ± {np.std(all_pearsons[:, :, i]):.2f}\n"
                )
            f.write(
                f"Average R2 for emotion responses: {mean_emo_r2:.2f} ± {std_emo_r2:.2f}\n"
            )
            f.write(
                f"Average Pearson's r for emotion responses: {mean_emo_pears:.2f} ± {std_emo_pears:.2f}\n"
            )
            f.write(
                f"Average R2 for mid-level responses: {mean_mid_r2:.2f} ± {std_mid_r2:.2f}\n"
            )
            f.write(
                f"Average Pearson's r for mid-level responses: {mean_mid_pears:.2f} ± {std_mid_pears:.2f}\n"
            )
