import numpy as np
import time
from scipy.sparse import issparse

from ..wrappers.sklearn_wrapper import OneClassSVMAnomalyDetector
from ..wrappers.sklearn_wrapper import IsolationForestAnomalyDetector
from ..wrappers.sklearn_wrapper import LocalOutlierFactorAnomalyDetector
from ..wrappers.keras_wrapper import SklearnKerasWrapper

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler


class AnomalyDetector(object):

    def __init__(self, detector_class, detector_opts, anomaly_class=None, features_number=0, epochs_number=0,
                 level=0, fold=0, n_clusters=1, optimize=False, weight_features=False, workers_number=1,
                 unsupervised=False, arbitrary_discr=''):

        self.anomaly_class = anomaly_class
        if self.anomaly_class == 1:
            self.normal_class = 0
        else:
            self.normal_class = 1
        if detector_class.startswith('k'):
            self.dl = True
        else:
            self.dl = False
        self.n_clusters = n_clusters
        self.optimize = optimize
        self.unsupervised = unsupervised
        self.anomaly_detectors = []
        for _ in range(self.n_clusters):
            if detector_class == 'ssv':
                if not self.optimize:
                    self.anomaly_detector = OneClassSVMAnomalyDetector(kernel='rbf')
                else:
                    # Parameters for Grid Search
                    anomaly_detector = OneClassSVMAnomalyDetector()
                    param_grid = {
                        'kernel': ['rbf', 'sigmoid'],
                        'gamma': np.linspace(.1, 1, 5),
                        'nu': np.linspace(.1, 1, 5)
                    }
                # [
                # ,
                # {
                #     'kernel': ['poly'],
                #     'degree': [3,5,7,9],
                #     'gamma': np.linspace(.1,1,10),
                #     'nu': np.linspace(.1,1,10)
                # },
                # {
                #     'kernel': ['linear'],
                #     'nu': np.linspace(.1,1,10)
                # }
                # ]
            elif detector_class == 'sif':
                if not self.optimize:
                    self.anomaly_detector = IsolationForestAnomalyDetector(n_estimators=100, contamination=0.)
                else:
                    # Parameters for Grid Search
                    anomaly_detector = IsolationForestAnomalyDetector()
                    param_grid = {
                        'n_estimators': [int(i) for i in np.linspace(10, 300, 5)],
                        'contamination': [0.]
                    }
                # self.grid_search_anomaly_detector = GridSearchCV(anomaly_detector, param_grid)
            elif detector_class == 'slo':
                if not self.optimize:
                    self.anomaly_detector = LocalOutlierFactorAnomalyDetector(
                        algorithm='ball_tree',
                        novelty=True,
                        n_neighbors=2,
                        leaf_size=100,
                        metric='canberra'
                    )
                else:
                    # Parameters for Grid Search
                    anomaly_detector = LocalOutlierFactorAnomalyDetector()
                    param_grid = [
                        {
                            'n_neighbors': [int(i) for i in np.linspace(2, 100, 5)],
                            'algorithm': ['kd_tree'],
                            'leaf_size': [int(i) for i in np.linspace(3, 150, 5)],
                            # 'metric': ['chebyshev', 'cityblock', 'euclidean', 'infinity', 'l1', 'l2', 'manhattan', 'minkowski', 'p'],
                            'contamination': [1e-4],
                            'novelty': [True]
                        },
                        {
                            'n_neighbors': [int(i) for i in np.linspace(2, 100, 5)],
                            'algorithm': ['ball_tree'],
                            'leaf_size': [int(i) for i in np.linspace(3, 150, 5)],
                            # 'metric': ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'dice', 'euclidean', 'hamming',\
                            # 'infinity', 'jaccard', 'kulsinski', 'l1', 'l2', 'manhattan', 'matching', 'minkowski', 'p',\
                            # 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath'],
                            'contamination': [1e-4],
                            'novelty': [True]
                        }
                        # ,
                        # {
                        #     'n_neighbors': [ int(i) for i in np.linspace(2,100,10) ],
                        #     'algorithm': ['brute'],
                        #     'metric': ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'cosine', 'dice',\
                        #     'euclidean', 'hamming', 'jaccard', 'kulsinski', 'l1', 'l2', 'manhattan', 'matching', 'minkowski',\
                        #     'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'],
                        #     'contamination': [1e-4],
                        #     'novelty': [True]
                        # }
                    ]
                # self.grid_search_anomaly_detector = GridSearchCV(anomaly_detector, param_grid)
            elif self.dl:
                if not self.optimize:
                    self.anomaly_detector = SklearnKerasWrapper(*detector_opts, model_class=detector_class,
                                                                num_classes=2, epochs_number=epochs_number, level=level,
                                                                fold=fold, weight_features=weight_features,
                                                                arbitrary_discr=arbitrary_discr)
                else:
                    # Parameters for Grid Search
                    anomaly_detector = SklearnKerasWrapper()
                    param_grid = [{
                        'sharing_ray': [str(i) for i in range(-1, d + 1)],
                        'grafting_depth': [str(i) for i in range(0, d + 2)],
                        'depth': [str(d)],
                        # 'compression_ratio': ['.1','.15','.2','.25'],
                        'compression_ratio': ['.1'],
                        # 'mode': ['n','v','s','p','sv','sp','vp','svp'],
                        'mode': ['n'],
                        # 'hidden_activation_function': ['elu','tanh','sigmoid'],
                        'hidden_activation_function_name': ['elu'],
                        'model_class': [detector_class],
                        'epochs_number': [epochs_number],
                        'num_classes': [2],
                        'fold': [fold],
                        'level': [level]
                        # } for d in range(1,6) ]
                    } for d in range(3, 4)]
                # self.grid_search_anomaly_detector = GridSearchCV(anomaly_detector, param_grid)
            if not self.optimize:
                self.anomaly_detectors.append(self.anomaly_detector)
        self.detector_class = detector_class

        if self.optimize:
            self.grid_search_anomaly_detector = GridSearchCV(anomaly_detector, param_grid, verbose=1,
                                                             n_jobs=workers_number)
            # self.grid_search_anomaly_detector.evaluate_candidates(ParameterGrid(param_grid))
            # input()
            self.file_out = 'data_%s/material/optimization_results_fold_%s.csv' % (detector_class, fold)

        # print(self.grid_search_anomaly_detector.get_params())
        # input()

    def fit(self, training_set, ground_truth):

        if not self.unsupervised:
            normal_index = [i for i, gt in enumerate(ground_truth) if gt == self.normal_class]
        else:
            normal_index = [i for i in range(len(ground_truth))]

        if self.dl:
            local_training_set = training_set[normal_index]
            clustering_training_set, self.train_scaler = self.scale_n_flatten(training_set[normal_index],
                                                                              return_scaler=True)
        else:
            local_training_set, self.train_scaler = self.scale_n_flatten(training_set[normal_index], return_scaler=True)
            clustering_training_set = local_training_set

        t = time.time()
        if self.n_clusters < 2:
            # if self.dl:
            if self.optimize:
                for row in local_training_set:
                    for elem in row:
                        try:
                            float(elem)
                        except:
                            print(row)
                            print(elem)
                            input()
                self.grid_search_anomaly_detector.fit(local_training_set, ground_truth[normal_index])
                self.anomaly_detector = self.grid_search_anomaly_detector.best_estimator_
                import pandas as pd
                df = pd.DataFrame(list(self.grid_search_anomaly_detector.cv_results_['params']))
                ranking = self.grid_search_anomaly_detector.cv_results_['rank_test_score']
                # The sorting is done based on the test_score of the models.
                sorting = np.argsort(self.grid_search_anomaly_detector.cv_results_['rank_test_score'])
                # Sort the lines based on the ranking of the models
                df_final = df.iloc[sorting]
                # The first line contains the best model and its parameters
                df_final.to_csv(self.file_out)
            else:
                self.anomaly_detector.fit(local_training_set, ground_truth[normal_index])
        # else:
        #     self.grid_search_anomaly_detector.fit(local_training_set, ground_truth[normal_index])
        #     self.anomaly_detector = self.grid_search_anomaly_detector.best_estimator_
        else:
            from sklearn.cluster import KMeans
            self.clustering = KMeans(n_clusters=self.n_clusters)
            # We compose training set to feed clustering, in particular categorical (nominal) features are converted from
            # array of 0 and 1 to string of 0 and 1, then numerical are treated as float and then minmax scaled
            # clustering_training_set, self.clustering_scaler = self.ndarray2str(training_set[normal_index], return_scaler=True)
            print('\nStarting clustering on training set\n')
            cluster_index = self.clustering.fit_predict(clustering_training_set)
            for i in range(self.n_clusters):
                print('\nTraining Model of Cluster n.%s\n' % (i + 1))
                red_normal_index = [j for j, v in enumerate(cluster_index) if v == i]
                self.anomaly_detectors[i].fit(local_training_set[red_normal_index], ground_truth[normal_index])

        t = time.time() - t

        self.t_ = t

        return self

    def predict(self, testing_set):

        # # Only for debugging
        # self.anomaly_detector.set_oracle(self.oracle)

        # self.clustering.transform(X) to get distances for each sample to each centers
        # self.clustering.cluster_centers_ to get centers

        if self.dl:
            local_testing_set = testing_set
            clustering_testing_set = self.scale_n_flatten(testing_set, self.train_scaler)
        else:
            local_testing_set = self.scale_n_flatten(testing_set, self.train_scaler)
            clustering_testing_set = local_testing_set

        if self.n_clusters < 2:
            pred = self.anomaly_detector.predict(local_testing_set)
            pred = np.asarray([self.anomaly_class if p < 0 else self.normal_class for p in pred])
        else:
            cluster_index = self.clustering.predict(clustering_testing_set)
            pred = np.ndarray(shape=[len(cluster_index), ])
            for i, _anomaly_detector in enumerate(self.anomaly_detectors):
                red_index = [j for j, v in enumerate(cluster_index) if v == i]
                pred[red_index] = _anomaly_detector.predict(local_testing_set[red_index])
                pred[red_index] = np.asarray(
                    [self.anomaly_class if p < 0 else self.normal_class for p in pred[red_index]])

        return pred

    def predict_proba(self, testing_set):

        if self.dl:
            local_testing_set = testing_set
            clustering_testing_set = self.scale_n_flatten(testing_set, self.train_scaler)
            score = self.anomaly_detector.predict_proba
            scores = [ad.predict_proba for ad in self.anomaly_detectors]
        else:
            local_testing_set = self.scale_n_flatten(testing_set, self.train_scaler)
            clustering_testing_set = local_testing_set
            score = self.anomaly_detector.decision_function
            scores = [ad.decision_function for ad in self.anomaly_detectors]

        if self.n_clusters < 2:
            proba = score(local_testing_set)
        else:
            cluster_index = self.clustering.predict(clustering_testing_set)
            proba = np.ndarray(shape=[len(cluster_index), ])
            for i, _anomaly_detector in enumerate(self.anomaly_detectors):
                red_index = [j for j, v in enumerate(cluster_index) if v == i]
                proba[red_index] = scores[i](local_testing_set[red_index])

        # We return the the opposite of decision_function because of the TPR and FPR we want to compute is related to outlier class
        # In this way, the higher is proba value, the more outlier is the sample.
        if not self.dl:
            proba = proba * -1

        return proba

    def set_oracle(self, oracle):

        self.oracle = oracle

        pass

    def get_distances(self, X, num_centers=None, nom_centers=None):
        '''
        This function take in input the matrix samples x features, where nominal are sting and numerical are float
        and centers vector, where 0-element refers numericals and 1 to nominals.
        Returns for each sample the distances from each centers, so a ndarray of shape samples x n_centroids
        '''
        num_index = [j for j, v in enumerate(X[0, :]) if isinstance(v, float)]
        nom_index = [j for j, v in enumerate(X[0, :]) if isinstance(v, str)]

        num_weight = len(num_index)
        nom_weight = len(nom_index)

        num_distances = []
        for num_centroid in num_centers:
            tmp_cen_dist = []
            for sample in X[:, num_index]:
                sample_dist = np.linalg.norm(num_centroid - sample)
                tmp_cen_dist.append(sample_dist)
            num_distances.append(tmp_cen_dist)
        num_distances = np.asarray(num_distances).T

        # np.asarray([ np.linalg.norm(X[:,num_index] - num_centroid, axis = 1) for num_centroid in num_centers ]).T

        nom_distances = []
        for nom_centroid in nom_centers:
            tmp_cen_dist = []
            for sample in X[:, nom_index]:
                sample_dist = np.sum(
                    [self.one_hot_hamming(nc, sm) for nc, sm in zip(nom_centroid, sample)]) / nom_weight
                tmp_cen_dist.append(sample_dist)
            nom_distances.append(tmp_cen_dist)
        nom_distances = np.asarray(nom_distances).T

        distances = np.average((num_distances, nom_distances), axis=0, weights=[num_weight, nom_weight])

        return distances

    def one_hot_hamming(self, u, v):
        '''
        Hamming distance of two string or vectors one hot encoded of equal length
        '''
        assert len(u) == len(v)
        if u.index('1') == v.index('1'):
            return 0.
        else:
            return 1.

    def ndarray2str(self, X, scaler=None, return_scaler=False):
        '''
        This function take in input a matrix samples x features, identifying nominal ones if they are ndarray, and transform them in string.
        Numerical features are minmaxscaled.
        Returns the entire X as a list of samples, where numerical features are float and nominal are str.
        '''
        if scaler is None:
            scaler = MinMaxScaler()
        samples_count = X.shape[0]
        trans_X = np.asarray([
            np.asarray([''.join([str(int(e)) for e in v]) for v in X[:, j]], dtype=object)
            if isinstance(v, np.ndarray)
            else scaler.fit_transform(np.asarray(X[:, j], dtype=float).reshape(-1, 1)).reshape((samples_count,))
            for j, v in enumerate(X[0, :])
        ], dtype=object).T

        if return_scaler:
            return trans_X, scaler
        return trans_X

    def scale_n_flatten(self, X, scaler=None, return_scaler=False):
        '''
        This function take in input a matrix samples x features, identifying nominal ones if they are ndarray,
        and transform them in flatten representation.
        Numerical features are minmaxscaled.
        Returns the entire X as a list of samples, where numerical features are float and nominal are str.
        '''
        if scaler is None:
            scaler = MinMaxScaler()
        samples_count = X.shape[0]

        trans_X = np.asarray([
            X[:, j]
            if issparse(v)
            else scaler.fit_transform(np.asarray(X[:, j], dtype=float).reshape(-1, 1)).reshape((samples_count,))
            for j, v in enumerate(X[0, :])
        ], dtype=object).T

        new_trans_X = []
        for i in range(trans_X.shape[0]):
            new_trans_X.append(np.hstack(trans_X[i]))
        trans_X = np.asarray(new_trans_X)

        if return_scaler:
            return trans_X, scaler
        return trans_X

    def fit_NeuralWeightsDecisor(self, training_set, ground_truth):

        decisor = AnomalyDetector(self.detector_class, self.detector_opts, self.anomaly_class, self.features_number,
                                  self.epochs_number,
                                  self.level, self.fold, self.n_clusters, self.optimize, self.weight_features,
                                  self.workers_number)

        self.fit(training_set, ground_truth)

        weights = self.decisor.get_weights()
