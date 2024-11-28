import os

import numpy as np
import pandas as pd
import scipy.stats
import torch
from scipy.io import arff
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

cwd = os.getcwd()

DATASET_NAMES = [
    "Gas Concentrations",
]


def add_bias(X: np.array) -> np.array:
    return np.insert(X, 0, 1, axis=1)


def get_datasets():
    names = DATASET_NAMES
    return {name: load_dataset(name) for name in names}


def load_dataset(name):
    if name == "Gas Concentrations":
        name = "gas-concentrations"

    if name in []:
        df = pd.read_csv(os.path.join(cwd, "data", f"{name}.csv"))
    else:
        data = arff.loadarff(os.path.join(cwd, "data", f"{name}.arff"))
        df = pd.DataFrame(data[0])

    if name == "gas-concentrations":
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
    else:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

    if name == "gas-concentrations":
        ethanol_ammonia_samples = np.where(np.isin(y, [b"1", b"3"]))[0]
        X = X.iloc[ethanol_ammonia_samples]
        y = y[ethanol_ammonia_samples]
        y = np.where(np.isin(y, [b"1"]), 1, 0)

    X = X.fillna(X.mean())

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X)

    return X, y


def get_dataset_stats():
    datasets = get_datasets()
    stats = []
    for dataset_name in datasets:
        X, y = datasets[dataset_name]
        stats.append(
            {
                "Name": dataset_name,
                "Number of samples": X.shape[0],
                "Number of features": X.shape[1],
                "\\alpha": np.round(np.mean(y == 1), 2),
            }
        )

    return pd.DataFrame.from_records(stats)


def gen_synthetic_dataset_M1(alpha, mu, N):
    pos_size = round(alpha * N)
    neg_size = N - pos_size
    positives = np.random.normal(mu, 1, pos_size)
    negatives = np.random.normal(0, 1, neg_size)

    df = pd.DataFrame(
        {
            "X1": np.concatenate([positives, negatives]),
        }
    )
    for i in range(2, 11):
        df[f"X{i}"] = np.random.normal(0, 1, N)
    df["y"] = np.concatenate([np.ones(pos_size), np.zeros(neg_size)])

    df = df.sample(frac=1)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return X, y


def gen_synthetic_dataset_M2(alpha, mu, N):
    pos_size = round(alpha * N)
    easy_pos_size = round(0.75 * pos_size)
    hard_pos_size = pos_size - easy_pos_size
    neg_size = N - pos_size

    easy_positives = np.random.normal(mu, 1, easy_pos_size)
    hard_positives = np.random.normal(0, 1, hard_pos_size)
    negatives = np.random.normal(0, 1, neg_size)

    df = pd.DataFrame(
        {
            "X1": np.concatenate([easy_positives, hard_positives, negatives]),
        }
    )
    for i in range(2, 11):
        df[f"X{i}"] = np.random.normal(0, 1, N)
    df["y"] = np.concatenate([np.ones(pos_size), np.zeros(neg_size)])

    df = df.sample(frac=1)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return X, y


def gen_M1_dataset(alpha=0.5, mu=1, N=5000):
    X, y = gen_synthetic_dataset_M1(alpha, mu, N)
    return X, y


def gen_M2_dataset(alpha=0.5, mu=1, N=5000):
    X, y = gen_synthetic_dataset_M2(alpha, mu, N)
    return X, y


def gen_probit_dataset(N, b, n_features=3, include_bias: bool = False):
    df = pd.DataFrame()
    for i in range(n_features):
        df[f"X{i+1}"] = np.random.normal(0, 1, N)

    X = df.to_numpy()
    if include_bias:
        X = add_bias(X)
    probit_probas = scipy.stats.norm.cdf(np.matmul(X, b))
    rand_res = np.random.random(N)
    df["y"] = np.where(probit_probas > rand_res, 1, 0)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return X, y


def gen_cauchy_dataset(N, b, n_features=3, include_bias: bool = False):
    df = pd.DataFrame()
    for i in range(n_features):
        df[f"X{i+1}"] = np.random.normal(0, 1, N)

    X = df.to_numpy()
    if include_bias:
        X = add_bias(X)
    cauchy_probas = scipy.stats.cauchy.cdf(np.matmul(X, b))
    rand_res = np.random.random(N)
    df["y"] = np.where(cauchy_probas > rand_res, 1, 0)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return X, y


def gen_trimmed_logit_dataset(
    N, b, n_features=3, include_bias: bool = False, limit=0.1
):
    df = pd.DataFrame()
    for i in range(n_features):
        df[f"X{i+1}"] = np.random.normal(0, 1, N)

    X = df.to_numpy()
    if include_bias:
        X = add_bias(X)

    logit_probas = scipy.stats.logistic.cdf(np.matmul(X, b))
    upper_lim = 1 - limit
    lower_lim = limit
    trimmed_probas = np.maximum(np.minimum(logit_probas, upper_lim), lower_lim)

    rand_res = np.random.random(N)
    df["y"] = np.where(trimmed_probas > rand_res, 1, 0)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return X, y


def create_s_sar_scenario_3(x, y, k=5, p_minus=0.2, p_plus=0.6):
    k = x.shape[1]
    propensity_elems = p_minus + (x - np.min(x, axis=0)) / (
        np.max(x, axis=0) - np.min(x, axis=0)
    ) * (p_plus - p_minus)
    propensity = np.prod(propensity_elems, axis=1) ** (1 / k)

    s = np.array(y)
    positives = np.where(y == 1)[0]
    n_y1 = len(positives)

    new_unlabeled_samples = positives[
        np.random.random(len(positives)) < 1 - propensity[positives]
    ]
    s[new_unlabeled_samples] = 0
    n_s0 = len(new_unlabeled_samples)

    n_s1 = n_y1 - n_s0
    real_c = n_s1 / n_y1
    return s, real_c


def create_s_sar_LBE(x, y, c):
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression()
    clf.fit(x, y)

    def eta(x, lgr_param, intercept, kappa=10):
        return torch.pow(
            1 / (1 + torch.exp(-(x.double() @ lgr_param.T + intercept))), kappa
        )

    kappa = 10

    x = torch.tensor(x.to_numpy())
    y = torch.tensor(y)

    propensity = (
        eta(
            x,
            torch.tensor(clf.coef_, dtype=torch.double),
            torch.tensor(clf.intercept_, dtype=torch.double),
            kappa=kappa,
        )
        .reshape(-1)
        .double()
    )
    propensity[torch.where(y == 0)] = 0
    propensity

    n_pos = torch.sum(y)
    num_labeled = int(c * n_pos)
    idx = propensity.multinomial(num_samples=num_labeled, replacement=True)
    s = torch.zeros_like(y)
    s[idx] = 1

    real_c = torch.sum(s) / torch.sum(y)

    return s.numpy(), real_c


def create_s_SCAR(x, y, c):
    y = torch.from_numpy(y)
    propensity = c * torch.ones_like(y)
    propensity[torch.where(y == 0)] = 0
    propensity

    s = torch.where(torch.rand_like(propensity) < propensity, 1, 0)
    real_c = torch.sum(s) / torch.sum(y)

    return s.numpy(), real_c


def preprocess(X, y, s, test_size=0.2, n_best_features=5):
    X = X.fillna(X.mean())
    y = np.array(y)

    if n_best_features is not None and n_best_features < X.shape[1]:
        mutual_info_scores = mutual_info_classif(X, s)
        mutual_info_best = np.argsort(mutual_info_scores)[::-1]
        X = X.iloc[:, mutual_info_best[:n_best_features]]

    if test_size != 0:
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            X, y, s, test_size=test_size
        )
    else:
        X_train, X_test, y_train, y_test, s_train, s_test = (
            X,
            np.array([]),
            y,
            np.array([]),
            s,
            np.array([]),
        )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    if test_size != 0:
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, s_train, s_test


def get_small_dataset(
    name,
    device,
    case_control,
    scar,
    label_frequency,
    val_size=0.15,
    test_size=0.15,
):
    X, y = load_dataset(name)

    if not scar:
        s, c = create_s_sar_LBE(X, y, label_frequency)
    else:
        s, c = create_s_SCAR(X, y, label_frequency)

    pi_p = np.mean(y == 1)
    label_frequency = np.mean(s == 1)
    n_input = X.shape[1]

    x_train, x_rest, y_train, y_rest, o_train, o_rest = preprocess(
        X, y, s, test_size=val_size + test_size, n_best_features=None
    )
    x_val, x_test, y_val, y_test, o_val, o_test = train_test_split(
        x_rest, y_rest, o_rest, test_size=test_size / (val_size + test_size)
    )

    if case_control:
        l_indices = np.where(o_train == 1)
        x_train = np.concatenate([x_train, x_train[l_indices]])
        y_train = np.concatenate([y_train, y_train[l_indices]])
        o_train = np.concatenate([torch.zeros_like(o_train), torch.ones(l_indices)])

        idx = np.random.shuffle(np.arange(len(x_train)))
        x_train = x_train[idx]
        y_train = y_train[idx]
        o_train = o_train[idx]

    return (
        (x_train, y_train, o_train),
        (x_val, y_val, o_val),
        (x_test, y_test, o_test),
        label_frequency,
        pi_p,
        n_input,
    )
