from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import issparse
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif


def preprocess_dataset(mode='classical'):
    if mode == 'classical':
        pass
    elif mode == 'other':
        pass


def ohe(X, nominal_features_index):
    features_encoder = OneHotEncoder()
    for i in nominal_features_index:
        X[:, i] = [sp for sp in features_encoder.fit_transform(X[:, i].reshape(-1, 1))]


def sparse_flattening(X):
    new_data = []
    for col in X.T:
        if issparse(col[0]):
            for col0 in np.array([c.todense() for c in col]).T:
                new_data.append(col0[0])
        else:
            new_data.append(col)
    return np.array(new_data).T


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
def apply_anomalies(X, target_index, anomaly_classes):
    '''
    :param X: data
    :param anomaly_classes:
    :param anomaly_index: self.dataset_features_number + self.level_target
    :return:
    '''
    for anomaly_class in anomaly_classes:
        X[np.where(X[:, target_index] == anomaly_class), target_index] = '1'
    X[np.where(X[:, target_index] != '1'), target_index] = '0'
    anomaly_class = '1'
    return anomaly_class


# TODO: manage hidden classes in presence of benign declared and array of hidden classes
def apply_benign_hiddens(X, target_index, benign_class, hidden_classes):
    '''
        :param X: data
        :param benign_class:
        :param hidden_classes:
        :param anomaly_index: self.dataset_features_number + self.level_target
        :return:
        '''
    if len(hidden_classes) > 0:
        X[np.where((X[:, target_index] != benign_class) & (X[:, target_index] != hidden_classes)), target_index] = '1'
    else:
        X[np.where(X[:, target_index] != benign_class), target_index] = '1'
    X[np.where(X[:, target_index] == benign_class), target_index] = '0'
    anomaly_class = '1'
    return anomaly_class


# NON APPLICARE LA CAZZO DI FEATURES SELECTION SUPERVISIONATA QUANDO FAI ANOMALY DETECTION
# PORCODDIO
def feature_selection(X, y, node, dataset_features_number):
    node.features_index = []
    # TODO: does not work with onehotencoded and with the new names of dataset
    if node.features_number != 0 and node.features_number < dataset_features_number:
        selector = SelectKBest(mutual_info_classif, k=node.features_number)
        selector.fit(
            X[node.train_index],
            node.label_encoder.transform(y[node.train_index, node.level])
        )
        support = selector.get_support()
        node.features_index = [i for i, v in enumerate(support) if v]
    else:
        if node.packets_number == 0:
            node.features_index = range(dataset_features_number)
        else:
            node.features_index = np.r_[0:node.packets_number,
                             dataset_features_number / 2:dataset_features_number / 2 + node.packets_number]
    return node.features_index
