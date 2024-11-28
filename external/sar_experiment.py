import threading
import os
import json
import time
import torch
import copy
import numpy as np
import tensorflow as tf

from sarpu.pu_learning import pu_learn_sar_em
from sarpu.experiments import evaluate_all

class BaseThreadedExperiment(threading.Thread):
    def __init__(self, train_samples, test_samples, experiment_id, c, config, method_dir, sem):
        threading.Thread.__init__(self)
        self.train_samples, self.test_samples = train_samples, test_samples
        self.experiment_id = experiment_id
        self.c = c
        self.config = copy.deepcopy(config)
        self.method_dir = method_dir
        self.sem = sem
    
    def run(self):
        self.sem.acquire()
        self.train_and_test(self.train_samples, self.test_samples)
        self.sem.release()
    
    def train_and_test(self, train_samples, test_samples):
        raise NotImplementedError()

class SAREMThreadedExperiment(BaseThreadedExperiment):
    def __init__(self, train_samples, test_samples, experiment_id, c, config, method_dir, sem):
        BaseThreadedExperiment.__init__(self, train_samples, test_samples, experiment_id, c, config, method_dir, sem)
    
    def train_and_test(self, train_samples, test_samples):
        np.random.seed(self.experiment_id)
        torch.manual_seed(self.experiment_id)
        tf.random.set_seed(self.experiment_id)
            
        X_train, y_train, s_train = train_samples
        X_test, y_test, s_test = test_samples

        y_train = np.where(y_train == 1, 1, 0)
        y_test = np.where(y_test == 1, 1, 0)

        em_training_start = time.perf_counter()

        f_model, e_model, info = pu_learn_sar_em(X_train, s_train, range(X_train.shape[1]), \
            verbose=True, log_prefix=f'Exp {self.experiment_id}, c: {self.c:.2f} || ')
        
        em_training_time = time.perf_counter() - em_training_start

        # evaluate
        propensity = np.zeros_like(s_test)
        metrics = evaluate_all(y_test, s_test, propensity, f_model.predict_proba(X_test), e_model.predict_proba(X_test))
        metrics['f_accuarcy'], metrics['f_precision'], metrics['f_recall'], metrics['f_f1']
        
        metric_values = {
            'Method': 'SAR-EM',
            'Accuracy': metrics['f_accuarcy'],
            'Precision': metrics['f_precision'],
            'Recall': metrics['f_recall'],
            'F1 score': metrics['f_f1'],
            'Time': em_training_time
        }

        os.makedirs(self.method_dir, exist_ok=True)
        with open(os.path.join(self.method_dir, 'metric_values.json'), 'w') as f:
            json.dump(metric_values, f)
