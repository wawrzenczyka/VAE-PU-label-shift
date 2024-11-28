import numpy as np
from sklearn import metrics


def calculate_metrics(
    y_proba,
    y,
    s,
    ls_s_proba,
    no_ls_s_proba,
    method=None,
    ls_method=None,
    ls_pi=None,
    time=None,
    augmented_label_shift=False,
    cutoff_label_shift=False,
    cutoff_true_pi_shift=False,
    odds_ratio_label_shift=False,
    non_ls_augmented=False,
    pi_train_true=None,
    pi_shift_true=None,
    pi_shift_estimation_simple=None,
    pi_shift_estimation_em=None,
    em_label_shift=False,
    em_label_shift_proba_function=None,
    simple_label_shift=False,
    simple_label_shift_proba_function=None,
):
    y = np.where(y == 1, 1, 0)
    s = np.where(s == 1, 1, 0)

    if augmented_label_shift:
        y_pred = np.where(y_proba * ls_s_proba / no_ls_s_proba > 0.5, 1, 0)
    elif em_label_shift:
        y_pred = np.where(em_label_shift_proba_function(y_proba) > 0.5, 1, 0)
    elif simple_label_shift:
        y_pred = np.where(simple_label_shift_proba_function(y_proba) > 0.5, 1, 0)
    elif cutoff_label_shift:
        d_B_PU = (y_proba - ls_s_proba) / (1 - y_proba)
        theta = (1 / pi_shift_estimation_simple - 1) / (1 / pi_train_true - 1)
        y_pred = np.where(d_B_PU > theta, 1, 0)
    elif cutoff_true_pi_shift:
        d_B_PU = (y_proba - ls_s_proba) / (1 - y_proba)
        theta = (1 / pi_shift_true - 1) / (1 / pi_train_true - 1)
        y_pred = np.where(d_B_PU > theta, 1, 0)
    elif non_ls_augmented:
        d_B_PU = (y_proba - ls_s_proba) / (1 - y_proba)
        theta = 1
        y_pred = np.where(d_B_PU > theta, 1, 0)
    elif odds_ratio_label_shift:
        t = (1 / pi_shift_estimation_simple - 1) / (1 / pi_train_true - 1)
        od_cutoff = t / (1 + t)
        y_pred = np.where(y_proba > od_cutoff, 1, 0)
    else:
        y_pred = np.where(y_proba > 0.5, 1, 0)

    y_pred = np.where(s == 1, 1, y_pred)  # augmented

    accuracy = metrics.accuracy_score(y, y_pred)
    precision = metrics.precision_score(y, y_pred)
    recall = metrics.recall_score(y, y_pred)
    f1 = metrics.f1_score(y, y_pred)
    auc = metrics.roc_auc_score(y, y_pred)
    balanced_accuracy = metrics.balanced_accuracy_score(y, y_pred)

    u_accuracy = metrics.accuracy_score(y[s == 0], y_pred[s == 0])
    u_precision = metrics.precision_score(y[s == 0], y_pred[s == 0])
    u_recall = metrics.recall_score(y[s == 0], y_pred[s == 0])
    u_f1 = metrics.f1_score(y[s == 0], y_pred[s == 0])
    u_auc = metrics.roc_auc_score(y[s == 0], y_pred[s == 0])
    u_balanced_accuracy = metrics.balanced_accuracy_score(y[s == 0], y_pred[s == 0])

    metric_values = {
        "Method": method,
        "Label shift method": ls_method,
        "Label shift \\pi": ls_pi,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 score": f1,
        "AUC": auc,
        "Balanced accuracy": balanced_accuracy,
        "U-Accuracy": u_accuracy,
        "U-Precision": u_precision,
        "U-Recall": u_recall,
        "U-F1 score": u_f1,
        "U-AUC": u_auc,
        "U-Balanced accuracy": u_balanced_accuracy,
        "True label shift \\pi": float(pi_shift_true),
        "Immediate \\pi estimation": (
            float(pi_shift_estimation_simple) if simple_label_shift else None
        ),
        "EM \\pi estimation": float(pi_shift_estimation_em) if em_label_shift else None,
        "Time": time,
    }
    return metric_values
