from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA
from scipy.sparse import issparse
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import scale


CLASSIC = 'classic'
CUSTOM = 'custom'
FLATTEN = 'flatten'
MLO = 'mlo'


def preprocess_dataset(X, nominal_features_index, mode=CLASSIC):
    if CLASSIC in mode:
        pass
    elif CUSTOM in mode:
        ohe(X, nominal_features_index)
    elif MLO in mode:
        # OrdinalEncoding
        oe(X, nominal_features_index)
        # Flattening eventual Packet-based Categorical feature (Hystograms)
        X = sparse_flattening(X)
        # TODO: subsequents pre-processing steps should be done on the sole train set
        # PCA
        X = pca(X, n_components=10)
        # Z-score
        X = z_score(X)
    if FLATTEN in mode:
        X = sparse_flattening(X)
    return X


def ohe(X, nominal_features_index, features_encoders=None):
    if features_encoders is None:
        features_encoders = []
        for i in nominal_features_index:
            features_encoder = OneHotEncoder()
            X[:, i] = [sp for sp in features_encoder.fit_transform(X[:, i].reshape(-1, 1))]
            features_encoders.append(features_encoder)
        return features_encoders
    else:
        for i, features_encoder in zip(nominal_features_index, features_encoders):
            X[:, i] = [sp for sp in features_encoder.transform(X[:, i].reshape(-1, 1))]


def sparse_flattening(X):
    new_data = []
    for col in X.T:
        if issparse(col[0]):
            for col0 in np.array([c.todense() for c in col]).T:
                new_data.append(col0[0])
        else:
            new_data.append(col)
    return np.array(new_data).T


def oe(X, nominal_features_index):
    features_encoder = OrdinalEncoder()
    if len(nominal_features_index) > 1:
        X[:, nominal_features_index] = features_encoder.fit_transform(X[:, nominal_features_index])
    else:
        X[:, nominal_features_index] = features_encoder.fit_transform(X[:, nominal_features_index].reshape(-1,1))


def pca(X, k_best=None, n_components=None):
    if k_best is not None:
        features_decomposer = PCA()
        features_decomposer.fit(X)
        variances = features_decomposer.explained_variance_
        X_decomp = X[:, sorted(list(reversed(np.argsort(variances)))[:k_best])]
    elif n_components is not None:
        features_decomposer = PCA(n_components)
        X_decomp = features_decomposer.fit_transform(X)
    return X_decomp


def z_score(X):
    return scale(X)


def get_num_nom_lengths(X):
    numerical_features_length = 0
    nominal_features_lengths = []
    for obj in X[0]:
        if issparse(obj):
            nominal_features_lengths.append(obj.shape[1])
        else:
            numerical_features_length += 1
    return numerical_features_length, nominal_features_lengths


# TODO: now it works only on one level dataset for AD, where when multiclass and anomaly is specified, anomaly goes to '1' and others to '0'
def apply_anomalies(y, target_index, anomaly_classes):
    '''
    :param X: data
    :param anomaly_classes:
    :param anomaly_index: self.dataset_features_number + self.level_target
    :return:
    '''
    for anomaly_class in anomaly_classes:
        y[np.where(y[:, target_index] == anomaly_class), target_index] = '1'
    y[np.where(y[:, target_index] != '1'), target_index] = '0'
    anomaly_class = '1'
    return anomaly_class


# TODO: manage hidden classes in presence of benign declared and array of hidden classes
def apply_benign_hiddens(y, target_index, benign_class, hidden_classes):
    '''
        :param X: data
        :param benign_class:
        :param hidden_classes:
        :param anomaly_index: self.dataset_features_number + self.level_target
        :return:
        '''
    if len(hidden_classes) > 0:
        y[np.where((y[:, target_index] != benign_class) & (y[:, target_index] != hidden_classes)), target_index] = '1'
    else:
        y[np.where(X[:, target_index] != benign_class), target_index] = '1'
    y[np.where(X[:, target_index] == benign_class), target_index] = '0'
    anomaly_class = '1'
    return anomaly_class


# NON APPLICARE LA CAZZO DI FEATURES SELECTION SUPERVISIONATA QUANDO FAI ANOMALY DETECTION
# PORCODDIO
def feature_selection(X, y, features_number, dataset_features_number):
    features_index = list(range(dataset_features_number))
    # TODO: does not work with onehotencoded and with the new names of dataset
    if features_number != 0 and features_number < dataset_features_number:
        selector = SelectKBest(mutual_info_classif, k=features_number)
        selector.fit(X, y)
        support = selector.get_support()
        features_index = [i for i, v in enumerate(support) if v]
    # else:
    #     if node.packets_number == 0:
    #         node.features_index = range(dataset_features_number)
    #     else:
    #         node.features_index = np.r_[0:node.packets_number,
    #                          dataset_features_number / 2:dataset_features_number / 2 + node.packets_number]
    return features_index
