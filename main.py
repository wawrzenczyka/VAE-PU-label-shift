# %%
import json
import multiprocessing
import os
import threading
import time

import numpy as np
import tensorflow as tf
import torch

from config import config
from data_loading.vae_pu_dataloaders import create_vae_pu_adapter, get_dataset
from external.LBE import eval_LBE, train_LBE
from external.nnPUlearning.api import nnPU
from external.sar_experiment import SAREMThreadedExperiment
from external.two_step import eval_2_step, train_2_step
from vae_pu_occ.vae_pu_occ_trainer import VaePuOccTrainer

label_frequencies = [
    # 0.9,
    # 0.7,
    0.5,
    # 0.3,
    # 0.1,
    # 0.02
]

start_idx = 0
num_experiments = 1
epoch_multiplier = 1

datasets = [
    "MNIST 3v5",
]

training_modes = [
    "VAE-PU",
]

label_shift_pis = [0.9, 0.7, 0.5, 0.3, 0.1]
label_shift_methods = [
    "None",
    "Augmented label shift",
    "Cutoff label shift",
    "Cutoff true pi label shift",
    "Odds ratio label shift",
    "EM label shift",
    "Simple label shift",
    "Non-LS augmented",
]

case_control = False
synthetic_labels = False

config["occ_methods"] = [
    # "IsolationForest",
    # "A^3",
    "OddsRatio-PUprop-e200-lr1e-4",
]

config["use_original_paper_code"] = False
config["use_old_models"] = True

config["vae_pu_variant"] = None
config["nnPU_beta"], config["nnPU_gamma"] = None, None

if config["nnPU_beta"] is not None and config["nnPU_gamma"] is not None:
    config["vae_pu_variant"] = (
        f"beta_{config['nnPU_beta']:.0e}_gamma_{config['nnPU_gamma']:.0e}"
    )

config["train_occ"] = True
config["occ_num_epoch"] = round(100 * epoch_multiplier)

config["early_stopping"] = True
config["early_stopping_epochs"] = 10

if config["use_original_paper_code"]:
    config["mode"] = "near_o"
else:
    config["mode"] = "near_y"

config["device"] = "auto"


if __name__ == "__main__":
    from argparse import ArgumentParser

    def none_or_float(value):
        if value == "None":
            return None
        return float(value)

    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        nargs="+",
        choices=[
            "MNIST 3v5",
            "MNIST OvE",
            "CIFAR CarTruck",
            "CIFAR MachineAnimal",
            "STL MachineAnimal",
            "CDC-Diabetes",
            "Synthetic SCAR",
            "MNIST 3v5 SCAR",
            "MNIST OvE SCAR",
            "CIFAR CarTruck SCAR",
            "CIFAR MachineAnimal SCAR",
            "STL MachineAnimal SCAR",
            "CDC-Diabetes SCAR",
        ],
        default=["MNIST 3v5"],
        required=False,
    )
    parser.add_argument(
        "-m",
        "--training_mode",
        type=str,
        nargs="+",
        choices=[
            "VAE-PU",
        ],
        required=False,
    )
    parser.add_argument("-c", "--c", type=float, nargs="+", required=False)
    parser.add_argument("-i", "--start_idx", type=int, required=False)
    parser.add_argument("-cc", "--case_control", action="store_true")
    parser.add_argument("-n", "--num_experiments", type=int, default=1, required=False)
    parser.add_argument(
        "-lsp", "--label_shift_pi", type=none_or_float, nargs="+", required=False
    )
    parser.add_argument(
        "-lsm",
        "--label_shift_method",
        type=str,
        nargs="+",
        required=False,
        choices=[
            "None",
            "Augmented label shift",
            "Cutoff label shift",
            "Cutoff true pi label shift",
            "Odds ratio label shift",
            "EM label shift",
            "Simple label shift",
            "Non-LS augmented",
        ],
    )
    parser.add_argument("--f", type=str, required=False)
    args = parser.parse_args()

    print("Arguments:")
    print(args)

    if args.dataset is not None:
        datasets = args.dataset
    if args.training_mode is not None:
        training_modes = args.training_mode
    if args.c is not None:
        label_frequencies = args.c
    if args.start_idx is not None:
        start_idx = args.start_idx
    if args.case_control is not None:
        case_control = args.case_control
    if args.num_experiments is not None:
        num_experiments = args.num_experiments
    if args.label_shift_pi is not None:
        label_shift_pis = args.label_shift_pi
    if args.label_shift_method is not None:
        label_shift_methods = args.label_shift_method

    print("Device:", "cuda" if torch.cuda.is_available() else "cpu")

# used by SAR-EM
n_threads = multiprocessing.cpu_count()
sem = threading.Semaphore(n_threads)
threads = []

config["label_shift_pis"] = label_shift_pis
config["label_shift_methods"] = label_shift_methods

for dataset in datasets:
    config["data"] = dataset

    if "SCAR" in config["data"]:
        config["use_SCAR"] = True
    else:
        config["use_SCAR"] = False

    for training_mode in training_modes:
        config["training_mode"] = training_mode
        for idx in range(start_idx, start_idx + num_experiments):
            for base_label_frequency in label_frequencies:
                config["base_label_frequency"] = base_label_frequency

                np.random.seed(idx)
                torch.manual_seed(idx)
                tf.random.set_seed(idx)

                if config["device"] == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"

                (
                    train_samples,
                    val_samples,
                    test_samples,
                    label_frequency,
                    pi_p,
                    n_input,
                ) = get_dataset(
                    config["data"],
                    device,
                    base_label_frequency,
                    use_scar_labeling=config["use_SCAR"],
                    case_control=case_control,
                    synthetic_labels=synthetic_labels,
                )
                vae_pu_data = create_vae_pu_adapter(
                    train_samples, val_samples, test_samples, device
                )

                config["label_frequency"] = label_frequency
                config["pi_p"] = pi_p
                config["pi_prime"] = 0.5
                config["n_input"] = n_input

                config["pi_pl"] = label_frequency * pi_p
                config["pi_pu"] = pi_p - config["pi_pl"]
                config["pi_u"] = 1 - config["pi_pl"]

                batch_size = 1000
                pl_batch_size = int(np.ceil(config["pi_pl"] * batch_size))
                u_batch_size = batch_size - pl_batch_size
                config["batch_size_l"], config["batch_size_u"] = (
                    pl_batch_size,
                    u_batch_size,
                )
                config["batch_size_l_pn"], config["batch_size_u_pn"] = (
                    pl_batch_size,
                    u_batch_size,
                )

                config["n_h_y"] = 10
                config["n_h_o"] = 2
                config["lr_pu"] = 3e-4
                config["lr_pn"] = 1e-5

                if config["data"] == "CDC-Diabetes":
                    epoch_multiplier = 0.5

                config["num_epoch_pre"] = round(100 * epoch_multiplier)
                config["num_epoch_step1"] = round(400 * epoch_multiplier)
                config["num_epoch_step_pn1"] = round(500 * epoch_multiplier)
                config["num_epoch_step_pn2"] = round(600 * epoch_multiplier)
                config["num_epoch_step2"] = round(500 * epoch_multiplier)
                config["num_epoch_step3"] = round(700 * epoch_multiplier)
                config["num_epoch"] = round(800 * epoch_multiplier)

                config["n_hidden_cl"] = []
                config["n_hidden_pn"] = [300, 300, 300, 300]

                if config["data"] == "MNIST OvE":
                    config["alpha_gen"] = 0.1
                    config["alpha_disc"] = 0.1
                    config["alpha_gen2"] = 3
                    config["alpha_disc2"] = 3
                elif config["data"] == "CDC-Diabetes":
                    config["alpha_gen"] = 0.01
                    config["alpha_disc"] = 0.01
                    config["alpha_gen2"] = 0.3
                    config["alpha_disc2"] = 0.3

                    config["n_h_y"] = 5

                    config["n_hidden_pn"] = [200, 200]
                    config["n_hidden_vae_e"] = [100, 100]
                    config["n_hidden_vae_d"] = [100, 100]
                    config["n_hidden_disc"] = [20]

                    config["lr_pu"] = 1e-5
                    config["lr_pn"] = 3e-5
                elif ("CIFAR" in config["data"] or "STL" in config["data"]) and config[
                    "use_SCAR"
                ]:
                    config["alpha_gen"] = 3
                    config["alpha_disc"] = 3
                    config["alpha_gen2"] = 1
                    config["alpha_disc2"] = 1
                    ### What is it?
                    config["alpha_test"] = 1.0
                elif (
                    "CIFAR" in config["data"] or "STL" in config["data"]
                ) and not config["use_SCAR"]:
                    config["alpha_gen"] = 0.3
                    config["alpha_disc"] = 0.3
                    config["alpha_gen2"] = 1
                    config["alpha_disc2"] = 1
                    ### What is it?
                    config["alpha_test"] = 1.0
                else:
                    config["alpha_gen"] = 1
                    config["alpha_disc"] = 1
                    config["alpha_gen2"] = 10
                    config["alpha_disc2"] = 10

                config["device"] = device
                config["directory"] = os.path.join(
                    "result",
                    config["data"],
                    str(base_label_frequency),
                    "Exp" + str(idx),
                )

                if "VAE-PU" in config["training_mode"]:
                    trainer = VaePuOccTrainer(
                        num_exp=idx,
                        model_config=config,
                        pretrain=True,
                        case_control=case_control,
                    )
                    trainer.train(vae_pu_data)
                else:
                    np.random.seed(idx)
                    torch.manual_seed(idx)
                    tf.random.set_seed(idx)
                    method_dir = os.path.join(
                        config["directory"], "external", config["training_mode"]
                    )

                    if config["training_mode"] == "SAR-EM":
                        exp_thread = SAREMThreadedExperiment(
                            train_samples,
                            test_samples,
                            idx,
                            base_label_frequency,
                            config,
                            method_dir,
                            sem,
                        )
                        exp_thread.start()
                        threads.append(exp_thread)
                    else:
                        if config["training_mode"] == "LBE":
                            log_prefix = f"Exp {idx}, c: {base_label_frequency} || "

                            lbe_training_start = time.perf_counter()
                            lbe = train_LBE(
                                train_samples,
                                val_samples,
                                verbose=True,
                                log_prefix=log_prefix,
                            )
                            lbe_training_time = time.perf_counter() - lbe_training_start

                            accuracy, precision, recall, f1 = eval_LBE(
                                lbe, test_samples, verbose=True, log_prefix=log_prefix
                            )
                        elif config["training_mode"] == "2-step":
                            training_start = time.perf_counter()
                            clf = train_2_step(train_samples, config, idx)
                            training_time = time.perf_counter() - training_start

                            accuracy, precision, recall, f1 = eval_2_step(
                                clf, test_samples
                            )
                        elif config["training_mode"] in ["nnPUss", "nnPU", "uPU"]:
                            x, y, s = train_samples
                            x = x.reshape(-1, 1, 28, 28)
                            x = (x + 1) / 2
                            y = y.astype(np.int64)
                            s = s.astype(np.int64)
                            train_samples = x, y, s

                            training_start = time.perf_counter()
                            clf = nnPU(model_name=config["training_mode"])
                            clf.train(train_samples, config["pi_p"])
                            training_time = time.perf_counter() - training_start

                            x, y, s = test_samples
                            x = x.reshape(-1, 1, 28, 28)
                            x = (x + 1) / 2
                            y = y.astype(np.int64)
                            s = s.astype(np.int64)
                            test_samples = x, y, s

                            accuracy, precision, recall, f1 = clf.evaluate(test_samples)

                        metric_values = {
                            "Method": config["training_mode"],
                            "Accuracy": accuracy,
                            "Precision": precision,
                            "Recall": recall,
                            "F1 score": f1,
                            "Time": training_time,
                        }

                        os.makedirs(method_dir, exist_ok=True)
                        with open(
                            os.path.join(method_dir, "metric_values.json"), "w"
                        ) as f:
                            json.dump(metric_values, f)

        for t in threads:
            t.join()

# %%
