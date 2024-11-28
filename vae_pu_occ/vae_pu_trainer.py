import copy
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pkbar
import tensorflow as tf
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, random_split

from data_loading.utils import InfiniteDataLoader
from metrics_custom import calculate_metrics
from vae_pu_occ.early_stopping import EarlyStopping

from .model import (
    VAEdecoder,
    VAEencoder,
    classifier_o,
    classifier_pn,
    discriminator,
    myPU,
)


class VaePuTrainer:
    def __init__(
        self,
        num_exp,
        model_config,
        pretrain=True,
        balanced_risk=False,
        balanced_cutoff=False,
        balanced_logit=False,
        balanced_savage=False,
        unbalanced_savage=False,
        case_control=False,
    ):
        self.num_exp = num_exp
        self.config = model_config
        self.use_original_paper_code = self.config["use_original_paper_code"]
        self.device = self.config["device"]
        self.pretrain = pretrain

        self.balanced_risk = balanced_risk
        self.balanced_cutoff = balanced_cutoff
        self.balanced_logit = balanced_logit
        self.balanced_savage = balanced_savage
        self.unbalanced_savage = unbalanced_savage
        self.case_control = case_control

        self.model_type = "VAE-PU"
        if self.case_control:
            self.model_type += "-CC"

        if self.balanced_risk:
            self.model_type += "-balanced-risk"
        elif self.balanced_cutoff:
            self.model_type += "-balanced-cutoff"
        elif self.balanced_logit:
            self.model_type += "-balanced-logit"
        elif self.unbalanced_savage:
            self.model_type += "-unbalanced-savage"
        elif self.balanced_savage:
            self.model_type += "-balanced-savage"

    def train(self, vae_pu_data):
        self._prepare_dataloaders(vae_pu_data)
        self._prepare_model()
        self._prepare_metrics()

        model_file = f"model_pre_occ_{self.model_type}.pt"
        trained_vae_pu_exists = os.path.exists(
            os.path.join(self.config["directory"], model_file)
        )
        if (
            self.config["use_old_models"]
            and not self.use_original_paper_code
            and self.config["vae_pu_variant"] is None
            and trained_vae_pu_exists
        ):
            self.model = self._load_trained_vae_pu()
        else:
            self.baseline_training_start = time.perf_counter()
            os.makedirs(self.config["directory"], exist_ok=True)

            if self.pretrain:
                self._pretrain_autoencoder()

            self.model.findPrior(self.x_pl_full, self.x_u_full)

            self.model_post_vae_training = None
            for epoch in range(self.config["num_epoch"]):
                start_time = time.time()
                epoch_losses = {
                    "ELBO": [],
                    "Adversarial generation": [],
                    "Discriminator": [],
                    "Label": [],
                    "Target classifier": [],
                }

                # training
                for x_pl, x_u in zip(self.DL_pl, self.DL_u):
                    x_pl, x_u = x_pl[0], x_u[0]

                    if epoch < self.config["num_epoch_step1"]:
                        # train autoencoder only, no label loss
                        self._train_autoencoder(epoch, x_pl, x_u, epoch_losses)
                    # num_epoch_step2 == num_epoch_step_pn1
                    elif epoch < self.config["num_epoch_step2"]:
                        # train target classifier only
                        self._train_target_classifier(epoch, x_pl, x_u, epoch_losses)
                    elif epoch < self.config["num_epoch_step_pn2"]:
                        # train autoencoder only, with label loss
                        self._train_autoencoder(epoch, x_pl, x_u, epoch_losses)
                    elif epoch < self.config["num_epoch_step3"]:
                        # train both
                        self._train_autoencoder(epoch, x_pl, x_u, epoch_losses)
                        self._train_target_classifier(epoch, x_pl, x_u, epoch_losses)
                    else:
                        # finish by training only target classifier
                        self._train_target_classifier(epoch, x_pl, x_u, epoch_losses)

                # metrics and results
                if epoch < self.config["num_epoch_step1"]:
                    # train autoencoder only, no label loss
                    if (epoch + 1) % 100 == 0:
                        self._check_discriminator_and_classifier(epoch, x_pl, x_u)
                    self._print_autoencoder_log(epoch, epoch_losses)
                    self._calculate_VAE_metrics()
                    self.timesAutoencoder.append(time.time() - start_time)
                elif epoch < self.config["num_epoch_step2"]:
                    # train target classifier only
                    if (epoch + 1) % 100 == 0:
                        self._check_discriminator_and_classifier(epoch, x_pl, x_u)
                    self._calculate_target_classifier_metrics(epoch, epoch_losses)
                    self.timesTargetClassifier.append(time.time() - start_time)
                elif epoch < self.config["num_epoch_step_pn2"]:
                    # train autoencoder only, with label loss
                    if (epoch + 1) % 100 == 0:
                        self._check_discriminator_and_classifier(epoch, x_pl, x_u)
                    self._print_autoencoder_log(epoch, epoch_losses)
                    self._calculate_VAE_metrics()
                    self.timesAutoencoder.append(time.time() - start_time)
                elif epoch < self.config["num_epoch_step3"]:
                    # train both
                    if (epoch + 1) % 100 == 0:
                        self._check_discriminator_and_classifier(epoch, x_pl, x_u)
                    self._print_autoencoder_log(epoch, epoch_losses)
                    self._calculate_target_classifier_metrics(epoch, epoch_losses)
                    self.timesAutoencoder.append(time.time() - start_time)
                    self.timesTargetClassifier.append(time.time() - start_time)
                else:
                    # finish by training only target classifier
                    self._calculate_target_classifier_metrics(epoch, epoch_losses)
                    self.timesTargetClassifier.append(time.time() - start_time)

                print(
                    f'Exp: {self.num_exp} / c = {self.config["base_label_frequency"]:.2f} / Epoch: {epoch + 1:4} |||| Remaining time (baseline): {(self.config["num_epoch"] - epoch) * (time.time() - start_time):.2f} sec'
                )
            self.baseline_training_time = (
                time.perf_counter() - self.baseline_training_start
            )

        if "Augmented label shift" in self.config["label_shift_methods"]:
            self.no_ls_s_model = self.train_no_ls_s_model()

        for label_shift_pi in self.config["label_shift_pis"]:
            for label_shift_method in self.config["label_shift_methods"]:
                np.random.seed(self.num_exp)
                torch.manual_seed(self.num_exp)
                tf.random.set_seed(self.num_exp)

                print(
                    f"--- Label shift method: {label_shift_method}, pi shift: "
                    + (
                        f"{label_shift_pi:.2f}"
                        if label_shift_pi is not None
                        else "None"
                    )
                    + " ---"
                )

                metric_values = self._calculate_ls_metrics(
                    method=self.model_type,
                    label_shift_method=label_shift_method,
                    label_shift_pi=label_shift_pi,
                    time=self.baseline_training_time,
                )
                self._save_final_vae_pu_metric_values(
                    metric_values, label_shift_method, label_shift_pi
                )
        return self.model

    def train_ls_s_model(self, ls_dataset):
        ls_train_dataset, ls_val_dataset = random_split(
            ls_dataset,
            [int(0.8 * len(ls_dataset)), len(ls_dataset) - int(0.8 * len(ls_dataset))],
        )
        ls_train_DL, ls_val_DL = DataLoader(
            ls_train_dataset,
            batch_size=128,
            shuffle=True,
            drop_last=True,
        ), DataLoader(
            ls_val_dataset,
            batch_size=128,
        )
        ls_s_model = self._train_custom_s_classifier(ls_train_DL, ls_val_DL)
        ls_s_model.eval()
        return ls_s_model

    def train_no_ls_s_model(self):
        if hasattr(self, "no_shift_s_model") and self.no_shift_s_model is not None:
            return self.no_shift_s_model

        DL_train = DataLoader(
            TensorDataset(
                torch.concat([self.x_pl_full, self.x_u_full]),
                torch.concat([self.y_pl_full, self.y_u_full]),
                torch.concat(
                    [
                        torch.ones(self.x_pl_full.shape[0]).to(self.x_pl_full.device),
                        torch.zeros(self.x_u_full.shape[0]).to(self.x_u_full.device),
                    ]
                ),
            ),
            batch_size=128,
            shuffle=True,
            drop_last=True,
        )
        DL_val = DataLoader(
            TensorDataset(self.x_val, self.y_val, self.s_val),
            batch_size=128,
        )
        self.no_shift_s_model = self._train_custom_s_classifier(DL_train, DL_val)
        self.no_shift_s_model.eval()
        return self.no_shift_s_model

    def fit_label_shift_EM(self, DL):
        y_probas = []

        for x, _, _ in DL:
            y_proba = self.model.model_pn.classify(x, sigmoid=True).reshape(-1)
            y_probas.append(y_proba)

        y_proba = torch.cat(y_probas).detach().cpu().numpy()

        pi_shift = 0.5
        pi = self.config["pi_p"]

        tol = 1e-5
        while True:
            pi_shift_prev = pi_shift
            P_ls = self.P_ls_EM(y_proba, pi_shift, pi)
            pi_shift = np.mean(P_ls)

            if np.abs(pi_shift_prev - pi) < tol:
                self.pi_shift_EM = pi_shift
                break
            else:
                tol += 1e-5

    def P_ls_EM(self, y_proba, pi_shift, pi):
        return ((pi_shift / pi) * y_proba) / (
            ((pi_shift / pi) * y_proba) + (((1 - pi_shift) / (1 - pi)) * (1 - y_proba))
        )

    def get_label_shift_proba_function(self, pi_shift):
        return lambda y_proba, pi_shift=pi_shift, pi=self.config["pi_pl"]: self.P_ls_EM(
            y_proba, pi_shift, pi
        )

    def _calculate_ls_metrics(self, method, label_shift_method, label_shift_pi, time):
        ls_dataset = self._get_label_shifted_test_dataset(label_shift_pi)
        ls_DL = DataLoader(
            ls_dataset,
            batch_size=self.config["batch_size_test"],
        )

        augmented_label_shift = "Augmented label shift" in label_shift_method
        cutoff_label_shift = "Cutoff label shift" in label_shift_method
        cutoff_true_pi_shift = "Cutoff true pi label shift" in label_shift_method
        odds_ratio_label_shift = "Odds ratio label shift" in label_shift_method
        em_label_shift = "EM label shift" in label_shift_method
        simple_label_shift = "Simple label shift" in label_shift_method
        non_ls_augmented = "Non-LS augmented" in label_shift_method

        y_probas = []
        y_trues = []
        s_trues = []
        ls_s_probas = []
        no_ls_s_probas = []

        ls_s_model = None
        if (
            augmented_label_shift
            or cutoff_label_shift
            or cutoff_true_pi_shift
            or non_ls_augmented
        ):
            ls_s_model = self.train_ls_s_model(ls_dataset)
        if em_label_shift:
            self.fit_label_shift_EM(DL=ls_DL)
        if (
            simple_label_shift
            or cutoff_label_shift
            or cutoff_true_pi_shift
            or odds_ratio_label_shift
        ):
            self.fit_label_shift_simple(DL=ls_DL)

        for x, y, s in ls_DL:
            y_proba = self.model.model_pn.classify(x, sigmoid=True).reshape(-1)
            y_probas.append(y_proba)
            y_trues.append(y)
            s_trues.append(s)

            if ls_s_model is not None:
                ls_s_proba = ls_s_model.classify(x, sigmoid=True).reshape(-1)
                ls_s_probas.append(ls_s_proba)
            if self.no_ls_s_model is not None:
                no_ls_s_proba = self.no_ls_s_model.classify(x, sigmoid=True).reshape(-1)
                no_ls_s_probas.append(no_ls_s_proba)

        y_proba = torch.cat(y_probas).detach().cpu().numpy()
        y_true = torch.cat(y_trues).detach().cpu().numpy()
        s_true = torch.cat(s_trues).detach().cpu().numpy()
        pi_shift_true = np.mean(y_true == 1)
        if ls_s_model is not None:
            ls_s_proba = torch.cat(ls_s_probas).detach().cpu().numpy()
        if self.no_ls_s_model is not None:
            no_ls_s_proba = torch.cat(no_ls_s_probas).detach().cpu().numpy()

        metric_values = calculate_metrics(
            y_proba,
            y_true,
            s_true,
            ls_s_proba if ls_s_model is not None else None,
            no_ls_s_proba if self.no_ls_s_model is not None else None,
            method=method,
            time=time,
            ls_method=label_shift_method,
            ls_pi=label_shift_pi,
            augmented_label_shift=augmented_label_shift,
            cutoff_label_shift=cutoff_label_shift,
            cutoff_true_pi_shift=cutoff_true_pi_shift,
            odds_ratio_label_shift=odds_ratio_label_shift,
            non_ls_augmented=non_ls_augmented,
            pi_train_true=self.config["pi_pl"],
            pi_shift_true=pi_shift_true,
            pi_shift_estimation_simple=(
                self.pi_shift_simple if hasattr(self, "pi_shift_simple") else None
            ),
            pi_shift_estimation_em=(
                self.pi_shift_EM if hasattr(self, "pi_shift_EM") else None
            ),
            em_label_shift=em_label_shift,
            em_label_shift_proba_function=self.get_label_shift_proba_function(
                self.pi_shift_EM if hasattr(self, "pi_shift_EM") else None
            ),
            simple_label_shift=simple_label_shift,
            simple_label_shift_proba_function=self.get_label_shift_proba_function(
                self.pi_shift_simple if hasattr(self, "pi_shift_simple") else None
            ),
        )

        return metric_values

    def fit_label_shift_simple(self, DL):
        s_shifts = []

        for _, _, s in DL:
            s_shifts.append(s)

        s_shift = torch.cat(s_shifts).detach().cpu().numpy()

        s_prior_shift = np.mean(s_shift == 1)
        s_prior = len(self.x_pl_full) / (len(self.x_pl_full) + len(self.x_u_full))

        pi = self.config["pi_p"]
        pi_shift = (s_prior_shift / s_prior) * pi
        self.pi_shift_simple = pi_shift

    def _prepare_metrics(self):
        self.elbos = []
        self.advGenerationLosses = []
        self.discLosses = []
        self.labelLosses = []
        self.targetClassifierLosses = []
        self.valAccuracies = []
        self.valLosses = []

        self.timesAutoencoder = []
        self.timesTargetClassifier = []

    def _save_final_vae_pu_metric_values(
        self, metric_values, label_shift_method, label_shift_pi
    ):
        metrics_path = os.path.join(
            self.config["directory"],
            f"metric_values_{self.model_type}_ls-{label_shift_method}-"
            + (f"{label_shift_pi:.2f}" if label_shift_pi is not None else "None")
            + ".json",
        )
        if self.use_original_paper_code:
            metrics_path = os.path.join(
                self.config["directory"], "metric_values_orig.json"
            )
        elif self.config["vae_pu_variant"] is not None:
            metrics_path = os.path.join(
                self.config["directory"],
                "variants",
                self.config["vae_pu_variant"],
                "metric_values.json",
            )
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

        with open(metrics_path, "w") as f:
            json.dump(metric_values, f)

        if not self.use_original_paper_code:
            with open(
                os.path.join(self.config["directory"], "settings.json"), "w"
            ) as f:
                json.dump(
                    {
                        "Label frequency": self.config["label_frequency"].item(),
                        "Pi": self.config["pi_p"].item(),
                        "True storey pi": (
                            self.config["pi_pu"] / self.config["pi_u"]
                        ).item(),
                    },
                    f,
                )

            model_file = f"model_pre_occ_{self.model_type}.pt"
            torch.save(self.model, os.path.join(self.config["directory"], model_file))

            log2 = open(os.path.join(self.config["directory"], "log_PN.txt"), "a")
            acc, precision, recall, f1_score, auc, b_acc = self.model.accuracy(
                self.DL_test,
                balanced_cutoff=self.balanced_cutoff,
                pi_p=self.config["pi_p"],
            )

            if self.config["train_occ"]:
                log2.write(
                    "final test pre-occ: acc: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}, f1-score: {3:.4f}".format(
                        acc, precision, recall, f1_score
                    )
                    + "\n"
                )
                print(
                    "final test pre-occ: acc: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}, f1-score: {3:.4f}".format(
                        acc, precision, recall, f1_score
                    )
                )
            else:
                log2.write(
                    "final test : acc: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}, f1-score: {3:.4f}".format(
                        acc, precision, recall, f1_score
                    )
                )
                print(
                    "final test : acc: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}, f1-score: {3:.4f}".format(
                        acc, precision, recall, f1_score
                    )
                )
            log2.close()

    def _train_custom_s_classifier(
        self, DL_train_s, DL_val_s, epochs=200, lr=1e-4, use_early_stopping=True
    ):
        self.model_s = classifier_pn(self.config).to(self.config["device"])
        opt_s = Adam(self.model_s.parameters(), lr=lr, eps=1e-07)
        criterion = BCEWithLogitsLoss()

        early_stopping = EarlyStopping(patience=10)

        for epoch in range(epochs):
            kbar = pkbar.Kbar(
                target=len(DL_train_s),
                epoch=epoch,
                num_epochs=epochs,
                width=8,
                always_stateful=False,
            )

            self.model_s.train()

            for i, (x, y, s) in enumerate(DL_train_s):
                opt_s.zero_grad()

                s_pred = self.model_s.classify(x, sigmoid=False).reshape(-1)
                loss = criterion(s_pred, s)

                loss.backward()
                opt_s.step()

                kbar.update(i, values=[("loss", loss.cpu().item())])

            self.model_s.eval()

            s_pred_no_sigm = []
            s_true = []
            for i, (x, y, s) in enumerate(DL_val_s):
                s_pred_no_sigm.append(
                    self.model_s.classify(x, sigmoid=False).reshape(-1)
                )
                s_true.append(s)
            s_pred_no_sigm = torch.cat(s_pred_no_sigm)
            s_true = torch.cat(s_true)
            s_pred = torch.sigmoid(s_pred_no_sigm)

            val_loss = criterion(s_pred_no_sigm, s_true)
            val_acc = torch.sum((s_pred > 0.5) == s_true) / len(s_true)

            kbar.add(
                1,
                values=[
                    ("val_loss", val_loss.cpu().item()),
                    ("val_acc", val_acc.cpu().item()),
                ],
            )

            if use_early_stopping and early_stopping.check_stop(
                epoch, val_loss, self.model_s
            ):
                return early_stopping.best_model

        if use_early_stopping:
            self.model_s = early_stopping.best_model
        return self.model_s

    def _calculate_target_classifier_metrics(self, epoch, losses):
        targetLoss = np.mean(losses["Target classifier"])
        print("epoch: {}, loss: {}".format(epoch + 1, targetLoss))

        val_acc, val_pr, val_re, val_f1, val_auc, val_b_acc = self.model.accuracy(
            self.DL_val,
            balanced_cutoff=self.balanced_cutoff,
            pi_p=self.config["pi_p"],
        )
        self.valAccuracies.append(val_acc)

        print(
            "...val: acc: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}, f1: {3:.4f}".format(
                val_acc, val_pr, val_re, val_f1
            )
        )

        val_loss = self.model.loss_val(self.x_val[:20], self.x_val[20:])
        self.valLosses.append(val_loss)
        print(val_loss)

        self.targetClassifierLosses.append(targetLoss)
        print(
            f'Exp: {self.num_exp} / c = {self.config["base_label_frequency"]:.2f} / Epoch: {epoch + 1:4} |||| PN loss: {targetLoss}'
        )

    def _calculate_VAE_metrics(self):
        val_acc, val_pr, val_re, val_f1, val_auc, val_b_acc = self.model.accuracy(
            self.DL_val,
            use_vae=True,
            balanced_cutoff=self.balanced_cutoff,
            pi_p=self.config["pi_p"],
        )

        print(
            "...(VAE) val: acc: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}, f1: {3:.4f}".format(
                val_acc, val_pr, val_re, val_f1
            )
        )

    def _print_autoencoder_log(self, epoch, losses):
        elbo, advGenLoss, discLoss, labelLoss = (
            np.mean(losses["ELBO"]),
            np.mean(losses["Adversarial generation"]),
            np.mean(losses["Discriminator"]),
            np.mean(losses["Label"]),
        )

        self.elbos.append(elbo)
        self.advGenerationLosses.append(advGenLoss)
        self.discLosses.append(discLoss)
        self.labelLosses.append(labelLoss)

        print(
            f'Exp: {self.num_exp} / c = {self.config["base_label_frequency"]:.2f} / Epoch: {epoch + 1:4} |||| ELBO loss: {elbo:.4f}, AdvGen loss: {advGenLoss:.4f}, Disc loss: {discLoss:.4f}, Label loss {labelLoss:.4f}'
        )

    def _check_discriminator_and_classifier(self, epoch, x_pl, x_u):
        d_x_pu, d_x_u = self.model.check_disc(x_pl, x_u)
        d_x_pu2, d_x_pl2 = self.model.check_pn(x_pl, x_u)

    def _train_target_classifier(self, epoch, x_pl, x_u, losses):
        if self.model_post_vae_training is None:
            print("Model saved post vae training!")
            self.model_post_vae_training = copy.deepcopy(self.model)

        if self.use_original_paper_code:
            l5 = self.model.train_step_pn(
                x_pl,
                x_u,
                self.balanced_risk,
                self.balanced_logit,
                self.balanced_savage,
                self.unbalanced_savage,
                self.case_control,
            )
        else:
            # use x_u_full (all U samples) instead of x_u
            l5 = self.model.train_step_pn(
                x_pl,
                self.x_u_full,
                self.balanced_risk,
                self.balanced_logit,
                self.balanced_savage,
                self.unbalanced_savage,
                self.case_control,
            )
        losses["Target classifier"].append(l5)
        return l5

    def _train_autoencoder(self, epoch, x_pl, x_u, losses):
        l3 = self.model.train_step_disc(x_pl, x_u)
        l1, l2, l4 = self.model.train_step_vae(x_pl, x_u, epoch)

        if np.isnan(l1):
            raise ValueError(f"Autoencoder loss was NaN in epoch {epoch + 1}")

        losses["ELBO"].append(l1)
        losses["Adversarial generation"].append(l2)
        losses["Discriminator"].append(l3)
        losses["Label"].append(l4)

        return l1, l2, l3, l4

    def _pretrain_autoencoder(self):
        self.preLoss1, self.preLoss2 = [], []

        pretrain_time = time.time()
        for epoch in range(self.config["num_epoch_pre"]):
            print(
                f'[PRE-TRAIN] Exp: {self.num_exp} / c = {self.config["base_label_frequency"]:.2f} / Epoch: {epoch + 1}'
            )

            start_time = time.time()
            lst_1 = []
            lst_2 = []

            for x_pl, x_u in zip(self.DL_pl, self.DL_u):
                x_pl, x_u = x_pl[0], x_u[0]
                l1 = self.model.pretrain(x_pl, x_u)
                if np.isnan(l1):
                    raise ValueError(
                        f"(PRETRAIN) Autoencoder loss was NaN in epoch {epoch + 1}"
                    )

                lst_1.append(l1)
                if self.config["bool_pn_pre"]:
                    l2 = self.model.train_step_pn_pre(x_pl, x_u)
                    lst_2.append(l2)

            self.preLoss1.append(sum(lst_1) / len(lst_1))
            if self.config["bool_pn_pre"]:
                self.preLoss2.append(sum(lst_2) / len(lst_2))

            end_time = time.time()
            print(
                "[PRE-TRAIN] Remaining time: {} sec".format(
                    (self.config["num_epoch_pre"] - epoch - 1) * (end_time - start_time)
                )
            )
            print(f"[PRE-TRAIN] VAE Loss: {sum(lst_1) / len(lst_1)}")
            if self.config["bool_pn_pre"]:
                print(f"[PRE-TRAIN] PN Loss: {sum(lst_2) / len(lst_2)}")

        print("PRE-TRAIN finish!")

    def _prepare_dataloaders(self, vae_pu_data):
        (
            self.x_pl_full,
            self.y_pl_full,
            self.x_u_full,
            self.y_u_full,
            self.x_val,
            self.y_val,
            self.s_val,
            self.x_test,
            self.y_test,
            self.s_test,
        ) = vae_pu_data

        if "MNIST" in self.config["data"]:
            self.x_pl_full = (self.x_pl_full + 1.0) / 2.0
            self.x_u_full = (self.x_u_full + 1.0) / 2.0
            self.x_val = (self.x_val + 1.0) / 2.0
            self.x_test = (self.x_test + 1.0) / 2.0

        self.DL_pl = InfiniteDataLoader(
            TensorDataset(self.x_pl_full),
            batch_size=self.config["batch_size_l"],
            shuffle=True,
        )
        self.DL_u = DataLoader(
            TensorDataset(self.x_u_full),
            batch_size=self.config["batch_size_u"],
            shuffle=True,
        )
        self.DL_u_full = DataLoader(
            TensorDataset(self.x_u_full, self.y_u_full),
            batch_size=self.config["batch_size_u"],
            shuffle=True,
        )
        self.DL_val = DataLoader(
            TensorDataset(self.x_val, self.y_val),
            batch_size=self.config["batch_size_val"],
            shuffle=True,
        )
        self.DL_test = DataLoader(
            TensorDataset(self.x_test, self.y_test),
            batch_size=self.config["batch_size_test"],
            shuffle=True,
        )

    def _prepare_model(self):
        model_en = VAEencoder(self.config).to(self.config["device"])
        model_de = VAEdecoder(self.config).to(self.config["device"])
        model_disc = discriminator(self.config).to(self.config["device"])
        model_cl = classifier_o(self.config).to(self.config["device"])
        model_pn = classifier_pn(self.config).to(self.config["device"])

        opt_en = Adam(model_en.parameters(), lr=self.config["lr_pu"], eps=1e-07)
        opt_de = Adam(model_de.parameters(), lr=self.config["lr_pu"], eps=1e-07)
        opt_disc = Adam(
            model_disc.parameters(),
            lr=self.config["lr_disc"],
            betas=(self.config["beta1"], self.config["beta2"]),
            eps=1e-07,
        )
        opt_cl = Adam(
            model_cl.parameters(), lr=self.config["lr_pu"], weight_decay=1e-5, eps=1e-07
        )
        opt_pn = Adam(
            model_pn.parameters(), lr=self.config["lr_pn"], weight_decay=1e-5, eps=1e-07
        )

        self.model = myPU(
            self.config,
            model_en,
            model_de,
            model_disc,
            model_cl,
            model_pn,
            opt_en,
            opt_de,
            opt_disc,
            opt_cl,
            opt_pn,
        )

    def _load_trained_vae_pu(self):
        model_file = f"model_pre_occ_{self.model_type}.pt"
        model = torch.load(os.path.join(self.config["directory"], model_file))

        if os.path.exists(
            os.path.join(
                self.config["directory"],
                f"metric_values_{self.model_type}.json",
            )
        ):
            with open(
                os.path.join(
                    self.config["directory"],
                    f"metric_values_{self.model_type}.json",
                ),
                "r",
            ) as f:
                metric_values = json.load(f)
        elif os.path.exists(
            os.path.join(
                self.config["directory"],
                f"metric_values_{self.model_type}_ls-None-0.50.json",
            )
        ):
            with open(
                os.path.join(
                    self.config["directory"],
                    f"metric_values_{self.model_type}_ls-None-0.50.json",
                ),
                "r",
            ) as f:
                metric_values = json.load(f)
        else:
            raise Exception("No metrics file")

        (
            self.acc_pre_occ,
            self.precision_pre_occ,
            self.recall_pre_occ,
            self.f1_pre_occ,
            self.auc_pre_occ,
            self.balanced_accuracy_pre_occ,
        ) = (
            metric_values["Accuracy"],
            metric_values["Precision"],
            metric_values["Recall"],
            metric_values["F1 score"],
            metric_values["AUC"],
            metric_values["Balanced accuracy"],
        )
        if "Time" in metric_values:
            self.baseline_training_time = metric_values["Time"]
        else:
            self.baseline_training_time = None
        print("Pre-OCC model loaded from file!")
        print(
            "final test pre-occ: acc: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}, f1-score: {3:.4f}".format(
                self.acc_pre_occ,
                self.precision_pre_occ,
                self.recall_pre_occ,
                self.f1_pre_occ,
            )
        )
        return model

    def _get_label_shifted_test_dataset(self, label_shift_pi):
        X, y, s = self.x_test, self.y_test, self.s_test
        if label_shift_pi is None:
            return TensorDataset(X, y, s)

        X_pos, y_pos, s_pos = X[y == 1], y[y == 1], s[y == 1]
        X_neg, y_neg, s_neg = X[y != 1], y[y != 1], s[y != 1]

        pi = torch.sum(y == 1) / len(y)
        if pi < label_shift_pi:
            n_pos = torch.sum(y == 1).item()
            n_neg = int(n_pos * (1 - label_shift_pi) / label_shift_pi)

            sampled_neg_idx = torch.ones(len(y_neg)).multinomial(
                num_samples=n_neg, replacement=True
            )

            X, y, s = (
                torch.concat([X_pos, X_neg[sampled_neg_idx]]),
                torch.concat([y_pos, y_neg[sampled_neg_idx]]),
                torch.concat([s_pos, s_neg[sampled_neg_idx]]),
            )
        elif pi > label_shift_pi:
            n_neg = torch.sum(y != 1)
            n_pos = int(n_neg * label_shift_pi / (1 - label_shift_pi))

            sampled_pos_idx = torch.ones(len(y_pos)).multinomial(
                num_samples=n_pos, replacement=True
            )

            X, y, s = (
                torch.concat([X_neg, X_pos[sampled_pos_idx]]),
                torch.concat([y_neg, y_pos[sampled_pos_idx]]),
                torch.concat([s_neg, s_pos[sampled_pos_idx]]),
            )

        return TensorDataset(X, y, s)
