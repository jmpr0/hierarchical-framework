import numpy as np
import time
from scipy.sparse import issparse
import random

from core.wrappers.sklearn_wrapper import OneClassSVMAnomalyDetector, IsolationForestAnomalyDetector, \
    LocalOutlierFactorAnomalyDetector
from ..wrappers.keras_wrapper import SklearnKerasWrapper
import core.model_selection.optimizers as optimizers

from sklearn.preprocessing import MinMaxScaler


class OptimizedModel(object):

    def __init__(self, model_class, level=0, fold=0, arbitrary_discr='', classify=False, modalities=None,
                 workers_number=None, num_classes=2):

        self.model_class = model_class
        self.multimodal = modalities is not None
        self.modalities = modalities

        model = None
        param_grid = None
        common_param = dict()

        if model_class == 'ssv':
            model = OneClassSVMAnomalyDetector
            param_grid = {
                'kernel': ['rbf', 'sigmoid'],
                'gamma': np.linspace(.1, 1, 5),
                'nu': np.linspace(.1, 1, 5)
            }
        elif model_class == 'sif':
            model = IsolationForestAnomalyDetector
            param_grid = {
                'n_estimators': [int(i) for i in np.linspace(10, 300, 5)]
            }
            common_param = {
                'contamination': 0.
            }
        elif model_class == 'slo':
            model = LocalOutlierFactorAnomalyDetector
            param_grid = {
                'n_neighbors': [int(i) for i in np.linspace(2, 100, 5)],
                'algorithm': ['kd_tree', 'ball_tree'],
                'leaf_size': [int(i) for i in np.linspace(3, 150, 5)]
            }
            common_param = {
                'contamination': 1e-4,
                'novelty': True
            }
        elif model_class == 'km2nn':
            model = SklearnKerasWrapper
            param_grid = {
                'n_hidden_layers': [n for n in range(1, 6)],
                'hidden_activation_function_name': ['relu', 'elu', 'sigmoid', 'tanh', 'selu', 'exponential'],
                'epochs_number': [e for e in range(10, 100, 10)],
                'patience': [p for p in range(1, 3)],
                'min_delta': [1e-6, 1e-5, 1e-4],
                'optimizer': ['adadelta', 'adam', 'sgd', 'adagrad'],
                'dropout_ratio': [d for d in np.arange(.1, 1., .1)]
            }
            common_param = {
                'model_class': model_class,
                'mode': 'n',
                'num_classes': num_classes,
                'fold': fold,
                'level': level,
                'classify': classify,
                'modalities': modalities,
                'arbitrary_discr': arbitrary_discr
            }
        self.model_optimizer = optimizers.Custom_GeneticAlgorithm(model, param_grid, verbose=1, n_jobs=workers_number,
                                                                  common_param=common_param)
        self.file_out = 'data_%s/material/%s_optimization_results_fold_%s.csv' % (model_class, arbitrary_discr, fold)

        self.model_ = None

    def fit(self, training_set, ground_truth):
        t = time.time()
        self.model_optimizer.fit(training_set, ground_truth)
        t = time.time() - t
        self.model_ = self.model_optimizer.best_estimator_
        import pandas as pd
        df = pd.DataFrame(list(self.model_optimizer.cv_results_['params']))
        ranking = self.model_optimizer.cv_results_['rank_test_score']
        # The sorting is done based on the test_score of the models.
        sorting = np.argsort(ranking)
        # Sort the lines based on the ranking of the models
        df_final = df.iloc[sorting]
        # The first line contains the best model and its parameters
        df_final.to_csv(self.file_out)
        self.tr_ = t

        return self
