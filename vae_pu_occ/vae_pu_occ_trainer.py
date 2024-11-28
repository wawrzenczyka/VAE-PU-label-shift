import copy
import json
import os
import time

import numpy as np
import pkbar
import tensorflow as tf
import torch
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from sklearn import metrics
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from external.A3_adapter import A3Adapter
from external.cccpv.methods import ConformalPvalues
from external.ecod_v2 import ECODv2
from external.em_pu_cc import train_em_pu_cc
from external.nnPUlearning.api import nnPU
from external.pyod_wrapper import PyODWrapper
from external.vpu.api import VPU
from vae_pu_occ.early_stopping import EarlyStopping
from vae_pu_occ.model import classifier_pn

from .vae_pu_trainer import VaePuTrainer


class VaePuOccTrainer(VaePuTrainer):
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
        super(VaePuOccTrainer, self).__init__(
            num_exp,
            model_config,
            pretrain,
            balanced_risk,
            balanced_cutoff,
            balanced_logit,
            balanced_savage,
            unbalanced_savage,
            case_control,
        )

    def train(self, vae_pu_data):
        super(VaePuOccTrainer, self).train(vae_pu_data)

        if (
            not self.config["train_occ"]
            or self.use_original_paper_code
            or self.config["vae_pu_variant"] is not None
        ):
            return

        self.modelOrig = copy.deepcopy(self.model)
        for occ_method in self.config["occ_methods"]:
            print("Starting", occ_method)
            self.occ_training_start = time.perf_counter()

            model = copy.deepcopy(self.modelOrig)
            np.random.seed(self.num_exp)
            torch.manual_seed(self.num_exp)
            tf.random.set_seed(self.num_exp)

            pi_pl, pi_u, pi_pu = (
                self.config["pi_pl"],
                self.config["pi_u"],
                self.config["pi_pu"],
            )
            pi_p, pi_prime = self.config["pi_p"], self.config["pi_prime"]

            pu_to_u_ratio = pi_pu / pi_u

            x_pu_gen = self._generate_x_pu()

            best_epoch = self.config["num_epoch"]
            best_f1 = 0
            best_acc = 0
            best_model = copy.deepcopy(model)
            early_stopping_counter = 0

            if "Clustering" in occ_method:
                true_x_pu, pu_indices = self._select_true_x_pu_clustering(
                    self.x_pl_full, self.x_u_full, pu_to_u_ratio, occ_method
                )
            elif "OddsRatio" in occ_method:
                if "e100" in occ_method:
                    e = 100
                elif "e50" in occ_method:
                    e = 50
                elif "e200" in occ_method:
                    e = 200

                if "lr1e-3" in occ_method:
                    lr = 1e-3
                elif "lr1e-2" in occ_method:
                    lr = 1e-2
                elif "lr1e-4" in occ_method:
                    lr = 1e-4
                elif "lr1e-5" in occ_method:
                    lr = 1e-5

                model_s = self._train_s_classifier(
                    epochs=e, lr=lr, use_early_stopping=("ES" in occ_method)
                )
                true_x_pu, pu_indices = self._select_true_x_pu_odds_ratio(
                    model_s,
                    self.x_u_full,
                    use_pu_proportion=("PUprop" in occ_method),
                    pu_to_u_ratio=pu_to_u_ratio,
                    cautious_k=(
                        1
                        if ("cautious-1" in occ_method)
                        else 2 if ("cautious-2" in occ_method) else None
                    ),
                )
            elif "MixupPU" in occ_method:
                x_l = torch.from_numpy(x_pu_gen)
                use_extra_penalty = False
                extra_penalty_config = {
                    "lambda_pi": 0.03,
                    "pi": pu_to_u_ratio,
                    "use_log": False,
                }

                if "+concat" in occ_method:
                    x_l = torch.concat(
                        [torch.from_numpy(x_pu_gen), self.x_pl_full.cpu()]
                    )
                if "+extra-loss" in occ_method:
                    use_extra_penalty = True
                    extra_penalty_config["lambda_pi"] = float(occ_method.split("-")[-1])

                    if "-log" in occ_method:
                        extra_penalty_config["use_log"] = True

                representation = "DV"
                if "NJW" in occ_method:
                    representation = "NJW"
                if "no-name" in occ_method:
                    representation = "no-name"

                normalize_phi = True
                if "no-norm" in occ_method:
                    normalize_phi = False

                self._train_mixup(
                    x_l,
                    self.x_u_full,
                    representation,
                    pu_to_u_ratio,
                    normalize_phi,
                    use_extra_penalty,
                    extra_penalty_config,
                )
                true_x_pu, pu_indices = self._select_true_x_pu_mixup(self.x_u_full)
            elif "EM-PU" in occ_method:
                self._train_em_pu()
                true_x_pu, pu_indices = self._select_true_x_pu_em_pu(
                    self.x_u_full, pu_to_u_ratio
                )
            elif "nnPU" in occ_method:
                self._train_nnPU()
                true_x_pu, pu_indices = self._select_true_x_pu_nnPU(
                    self.x_u_full, pu_to_u_ratio
                )
            else:
                self._train_occ(occ_method, x_pu_gen)
                true_x_pu, pu_indices = self._select_true_x_pu_occ(
                    self.x_u_full, pu_to_u_ratio
                )

            for epoch in range(
                self.config["num_epoch"],
                self.config["num_epoch"] + self.config["occ_num_epoch"],
            ):
                start_time = time.time()
                targetLosses = []

                targetClassifierLoss = self.model.train_step_pn_true_x_pu(
                    self.x_pl_full,
                    self.x_u_full,
                    true_x_pu,
                    pi_pl,
                    pi_u,
                    pi_pu,
                    pi_p,
                    pi_prime,
                    self.balanced_risk,
                    self.balanced_logit,
                    self.balanced_savage,
                    self.unbalanced_savage,
                    self.case_control,
                )
                targetLosses.append(targetClassifierLoss)

                occ_metrics, val_metrics = self._calculate_occ_metrics(
                    epoch, occ_method, pu_indices, targetLosses
                )
                val_acc, _, _, val_f1 = val_metrics

                # best f1 early stopping
                if val_f1 >= best_f1:
                    best_f1 = val_f1
                    best_acc = val_acc
                    best_model = copy.deepcopy(model)
                    best_epoch = epoch
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if (
                        self.config["early_stopping"]
                        and early_stopping_counter
                        == self.config["early_stopping_epochs"]
                    ):
                        model = copy.deepcopy(best_model)
                        val_f1 = best_f1
                        val_acc = best_acc
                        occ_metrics, val_metrics = self._calculate_occ_metrics(
                            epoch, occ_method, pu_indices, targetLosses
                        )

                        print(
                            f"Early stopping | Best model epoch: {best_epoch + 1}, f1: {best_f1:.4f}, acc: {best_acc:.4f}"
                        )
                        print("")
                        break

            self.occ_training_time = time.perf_counter() - self.occ_training_start
            self._save_vae_pu_occ_metrics(occ_method, occ_metrics, best_epoch)

        # # OCC training end

        if not self.use_original_paper_code:
            self._save_results()
            self._save_final_metrics()

        return model

    def _generate_x_pu(self):
        x_pus = []

        for x_pl, full_u in zip(self.DL_pl, self.DL_u_full):
            (x_u, _) = full_u
            x_pl = x_pl[0]

            # generate number of samples >= the whole dataset
            generated_batches = np.ceil(1 / self.config["pi_pl"]).astype(int)

            for _ in range(generated_batches):
                if self.use_original_paper_code:
                    _, _, _, x_pu, _ = self.model.generate(
                        x_pl, x_u, self.config["mode"]
                    )
                else:
                    # use x_u_full (whole U set) instead of x_u (batch only)
                    _, _, _, x_pu, _ = self.model.generate(
                        x_pl, self.x_u_full, self.config["mode"]
                    )
                x_pus.append(x_pu.detach())

        x_pu = torch.cat(x_pus)
        x_pu = x_pu.cpu().numpy()

        return x_pu

    def _train_s_classifier(
        self, batch_size=128, epochs=100, lr=1e-3, use_early_stopping=False
    ):
        model_s = classifier_pn(self.config).to(self.config["device"])
        opt_s = Adam(model_s.parameters(), lr=lr, eps=1e-07)
        criterion = BCEWithLogitsLoss()

        DL_train_s = DataLoader(
            TensorDataset(
                torch.cat([self.x_pl_full, self.x_u_full]),
                torch.cat(
                    [torch.ones(len(self.x_pl_full)), torch.zeros(len(self.x_u_full))]
                ).to(self.config["device"]),
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        DL_val_s = DataLoader(
            TensorDataset(self.x_val, self.s_val),
            batch_size=self.config["batch_size_val"],
            shuffle=True,
        )

        early_stopping = EarlyStopping(patience=10)

        for epoch in range(epochs):
            kbar = pkbar.Kbar(
                target=len(DL_train_s),
                epoch=epoch,
                num_epochs=epochs,
                width=8,
                always_stateful=False,
            )

            model_s.train()

            for i, (x, s) in enumerate(DL_train_s):
                opt_s.zero_grad()

                s_pred = model_s.classify(x, sigmoid=False).reshape(-1)
                loss = criterion(s_pred, s)

                loss.backward()
                opt_s.step()

                kbar.update(i, values=[("loss", loss.cpu().item())])

            model_s.eval()

            s_pred_no_sigm = []
            s_true = []
            for i, (x, s) in enumerate(DL_val_s):
                s_pred_no_sigm.append(model_s.classify(x, sigmoid=False).reshape(-1))
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
                epoch, val_loss, model_s
            ):
                return early_stopping.best_model

        if use_early_stopping:
            model_s = early_stopping.best_model
        return model_s

    def _select_true_x_pu_odds_ratio(
        self, model_s, x_u, use_pu_proportion=False, pu_to_u_ratio=None, cautious_k=None
    ):
        s_u_pred = model_s.classify(x_u, sigmoid=True)
        y_u_pred = self.model.model_pn.classify(x_u, sigmoid=True)

        odds_ratio = (y_u_pred - s_u_pred) / (1 - y_u_pred)
        odds_ratio = odds_ratio.reshape(-1)

        if use_pu_proportion:
            # Order approach
            sorted_indices = torch.argsort(odds_ratio.reshape(-1), descending=True)
            n_pu_samples = round(pu_to_u_ratio * len(x_u))
            if cautious_k is not None:
                k = cautious_k
                n = len(x_u)
                pi = self.config["pi_p"]

                n_pu_samples -= k * np.sqrt(n * pi * (1 - pi))
                n_pu_samples = max(int(np.round(n_pu_samples)), 1)

            pu_indices = sorted_indices[:n_pu_samples]
        else:
            pu_indices = torch.where(odds_ratio >= 1)[0]

        true_x_pu = x_u[pu_indices]
        return true_x_pu, pu_indices

    def _train_mixup(
        self,
        x_l,
        x_u,
        representation,
        pu_to_u_ratio,
        normalize_phi,
        use_extra_penalty,
        extra_penalty_config,
    ):
        self.mixup_model = VPU(
            representation=representation,
            pi=pu_to_u_ratio,
            normalize_phi=normalize_phi,
            use_extra_penalty=use_extra_penalty,
            extra_penalty_config=extra_penalty_config,
        )
        self.mixup_model.train(
            x_l=x_l.detach().cpu(),
            x_u=x_u.detach().cpu(),
        )

    def _select_true_x_pu_mixup(self, x_u):
        y_pred = torch.tensor(
            self.mixup_model.predict(x_u.detach().cpu()), device=self.config["device"]
        )
        pu_indices = torch.where(y_pred == 1)
        true_x_pu = x_u[pu_indices]
        return true_x_pu, pu_indices

    def _train_em_pu(self):
        self.em_pu_model = train_em_pu_cc(
            (
                torch.concat(
                    [
                        self.x_pl_full,
                        self.x_u_full,
                    ]
                )
                .cpu()
                .numpy(),
                None,
                torch.concat(
                    [
                        torch.ones(len(self.x_pl_full)),
                        torch.zeros(len(self.x_u_full)),
                    ]
                )
                .cpu()
                .numpy(),
            ),
            self.config,
            0,
        )

    def _select_true_x_pu_em_pu(self, x_u, pu_to_u_ratio):
        y_proba = torch.tensor(
            self.em_pu_model.predict_proba(x_u.cpu().numpy()),
            device=self.config["device"],
        )

        sorted_indices = torch.argsort(y_proba, descending=True)
        n_pu_samples = round(pu_to_u_ratio * len(x_u))
        pu_indices = sorted_indices[:n_pu_samples]
        true_x_pu = x_u[pu_indices]
        return true_x_pu, pu_indices

    def _train_nnPU(self):
        pu_to_u_ratio = self.config["pi_pu"] / self.config["pi_u"]

        self.nnPU_model = nnPU(model_name="nnPU")
        self.nnPU_model.train(
            (
                torch.concat(
                    [
                        self.x_pl_full,
                        self.x_u_full,
                    ]
                )
                .cpu()
                .numpy()
                .reshape(-1, 1, 28, 28),
                None,
                torch.concat(
                    [
                        torch.ones(len(self.x_pl_full)),
                        torch.zeros(len(self.x_u_full)),
                    ]
                )
                .cpu()
                .numpy(),
            ),
            pu_to_u_ratio,
        )

    def _select_true_x_pu_nnPU(self, x_u, pu_to_u_ratio):
        y_proba = torch.tensor(
            self.nnPU_model.predict_scores(
                x_u.cpu().numpy().reshape(-1, 1, 28, 28)
            ).reshape(-1),
            device=self.config["device"],
        )

        sorted_indices = torch.argsort(y_proba, descending=True)
        n_pu_samples = round(pu_to_u_ratio * len(x_u))
        pu_indices = sorted_indices[:n_pu_samples]
        true_x_pu = x_u[pu_indices]
        return true_x_pu, pu_indices

    def _train_occ(self, occ_method, x_pu_gen):
        # Initialize the one-class classifier
        contamination = 0.001
        contamination = min(max(contamination, 0.004), 0.1)
        if "OC-SVM" in occ_method:
            occ = OneClassSVM(nu=contamination, kernel="rbf", gamma=0.1)
        elif "IsolationForest" in occ_method:
            occ = IsolationForest(
                random_state=self.num_exp, contamination=contamination
            )
        elif "LocalOutlierFactor" in occ_method:
            occ = LocalOutlierFactor(novelty=True, contamination=contamination)
        elif "A^3" in occ_method:
            occ = A3Adapter(target_epochs=10, a3_epochs=10)
        else:
            if "ECOD" in occ_method:
                occ = PyODWrapper(ECOD(contamination=contamination, n_jobs=-2))
            if "ECODv2" in occ_method:
                occ = PyODWrapper(ECODv2(contamination=contamination, n_jobs=-2))
            elif "COPOD" in occ_method:
                occ = PyODWrapper(COPOD(contamination=contamination, n_jobs=-2))

        self.cc = ConformalPvalues(
            x_pu_gen, occ, calib_size=0.5, random_state=self.num_exp
        )
        return self.cc

    def _select_true_x_pu_occ(self, x_u, pu_to_u_ratio):
        pvals_one_class = self.cc.predict(x_u.cpu().numpy(), delta=0.05, simes_kden=2)
        pvals = pvals_one_class["Marginal"]
        pvals = torch.from_numpy(pvals).to(self.config["device"])

        # Order approach
        sorted_indices = torch.argsort(pvals, descending=True)
        n_pu_samples = round(pu_to_u_ratio * len(x_u))
        pu_indices = sorted_indices[:n_pu_samples]
        true_x_pu = x_u[pu_indices]
        return true_x_pu, pu_indices

    def _select_true_x_pu_clustering(self, x_pl, x_u, pu_to_u_ratio, occ_method):
        if "QuantMean" in occ_method:
            clustering_metric = "quantmean"
        elif "Mean" in occ_method:
            clustering_metric = "mean"
        elif "Quantile" in occ_method:
            clustering_metric = "quantile"
        else:
            raise NotImplementedError()

        best_distances, best_xus, best_pu_indices = (
            torch.empty(0).to(self.config["device"]),
            torch.empty((0, x_u.shape[1])).to(self.config["device"]),
            torch.empty(0, dtype=torch.int).to(self.config["device"]),
        )
        n_pu_samples = int(round(pu_to_u_ratio * len(x_u)))

        batch_size = self.config["batch_size_u"]

        for batch_start in range(0, len(x_u), batch_size):
            batch_end = min(batch_start + batch_size, len(x_u))

            x_u_batch = x_u[batch_start:batch_end]
            (
                best_xus,
                best_pu_indices,
                best_distances,
            ) = self.model.get_pu_from_clustering_batched(
                x_pl,
                x_u_batch,
                n_pu_samples,
                clustering_metric,
                batch_start,
                best_distances=best_distances,
                best_xus=best_xus,
                best_pu_indices=best_pu_indices,
            )
        true_x_pu = best_xus
        pu_indices = best_pu_indices
        return true_x_pu, pu_indices

    def _calculate_occ_metrics(self, epoch, occ_method, pu_indices, epochTargetLosses):
        y_u_pred = torch.zeros_like(self.y_u_full)
        y_u_pred[pu_indices] = 1

        y_u_cpu = self.y_u_full.detach().cpu().numpy()
        y_u_cpu = np.where(y_u_cpu == 1, 1, 0)
        y_u_pred_cpu = y_u_pred.detach().cpu().numpy()

        occ_acc = metrics.accuracy_score(y_u_cpu, y_u_pred_cpu)
        occ_auc = metrics.roc_auc_score(y_u_cpu, y_u_pred_cpu)
        occ_precision = metrics.precision_score(y_u_cpu, y_u_pred_cpu)
        occ_recall = metrics.recall_score(y_u_cpu, y_u_pred_cpu)
        occ_f1 = metrics.f1_score(y_u_cpu, y_u_pred_cpu)
        occ_fdr = np.sum((y_u_pred_cpu == 1) & (y_u_cpu == 0)) / np.sum(
            y_u_pred_cpu == 1
        )

        print(
            "OCC ({}) - epoch: {}, loss: {}".format(
                occ_method, epoch + 1, np.mean(epochTargetLosses)
            )
        )
        val_acc, val_pr, val_re, val_f1, val_auc, val_b_acc = self.model.accuracy(
            self.DL_val,
            balanced_cutoff=self.balanced_cutoff,
            pi_p=self.config["pi_p"],
        )
        self.valAccuracies.append(val_acc)
        print(
            "...val: acc: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}, f1 score: {3:.4f}".format(
                val_acc, val_pr, val_re, val_f1
            )
        )

        val_loss = self.model.loss_val(self.x_val[:20], self.x_val[20:])
        self.valLosses.append(val_loss)

        self.targetClassifierLosses.append(np.mean(epochTargetLosses))
        print(
            f'Exp: {self.num_exp} / c = {self.config["base_label_frequency"]:.2f} / Epoch: {epoch + 1:4} |||| PN loss: {np.mean(epochTargetLosses)}'
        )

        print(
            f'Exp: {self.num_exp} / c = {self.config["base_label_frequency"]:.2f} / Epoch: {epoch + 1:4} |||| OCC training time: {(time.perf_counter() - self.occ_training_start):.2f} sec'
        )
        return (occ_acc, occ_auc, occ_precision, occ_recall, occ_f1, occ_fdr), (
            val_acc,
            val_pr,
            val_re,
            val_f1,
        )

    def _save_vae_pu_occ_metrics(self, occ_method, occ_metrics, best_epoch):
        os.makedirs(
            os.path.join(self.config["directory"], "occ", occ_method), exist_ok=True
        )

        if (
            "Augmented label shift" in self.config["label_shift_methods"]
            and not hasattr(self, "no_ls_s_model")
            and self.no_ls_s_model is None
        ):
            self.no_ls_s_model = self.train_no_ls_s_model()

        for label_shift_pi in self.config["label_shift_pis"]:
            for label_shift_method in self.config["label_shift_methods"]:
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
                    method=f"{self.model_type}+{occ_method}",
                    label_shift_method=label_shift_method,
                    label_shift_pi=label_shift_pi,
                    time=self.occ_training_time,
                )

                if self.baseline_training_time is not None:
                    metric_values["Time"] = (
                        self.baseline_training_time + self.occ_training_time
                    )

                with open(
                    os.path.join(
                        self.config["directory"],
                        "occ",
                        occ_method,
                        f"metric_values_{self.model_type}+{occ_method}_ls-{label_shift_method}-"
                        + (
                            f"{label_shift_pi:.2f}"
                            if label_shift_pi is not None
                            else "None"
                        )
                        + ".json",
                    ),
                    "w",
                ) as f:
                    json.dump(metric_values, f)
        return self.model

    def _save_results(self):
        if len(self.timesAutoencoder) > 0 and len(self.timesTargetClassifier) > 0:
            print(
                np.mean(np.array(self.timesAutoencoder[1:])),
                np.mean(np.array(self.timesTargetClassifier[1:])),
            )

    def _save_final_metrics(self):
        acc, precision, recall, f1_score, auc, b_acc = self.model.accuracy(
            self.DL_test,
            balanced_cutoff=self.balanced_cutoff,
            pi_p=self.config["pi_p"],
        )

        if self.config["train_occ"] and hasattr(self, "acc_pre_occ"):
            print(
                "final test pre-occ: acc: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}, f1-score: {3:.4f}".format(
                    self.acc_pre_occ,
                    self.precision_pre_occ,
                    self.recall_pre_occ,
                    self.f1_pre_occ,
                )
            )

        print(
            "final test : acc: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}, f1-score: {3:.4f}".format(
                acc, precision, recall, f1_score
            )
        )

        # if self.config["train_occ"]:
        #     torch.save(
        #         self.model,
        #         os.path.join(self.config["directory"], f"model_{self.model_type}.pt"),
        #     )
