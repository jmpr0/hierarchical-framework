#!/usr/bin/python3

import os
import pickle as pk
from threading import Thread
import time

import arff
import networkx as nx
import numpy as np
from scipy.sparse import issparse

import warnings

from sklearn.preprocessing import OneHotEncoder
from .utils.encoders import MyLabelEncoder
from configs import *
from sklearn.model_selection import StratifiedKFold
from .utils.partitioners import equal_partitioner
from .models.anomaly_detector import AnomalyDetector
from .ensemble.trainable import SuperLearnerClassifier
from .wrappers.spark_wrapper import SklearnSparkWrapper
from .wrappers.weka_wrapper import SklearnWekaWrapper
from .wrappers.keras_wrapper import SklearnKerasWrapper
from sklearn.multiclass import OutputCodeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from .utils.preprocessing import apply_anomalies
from .utils.preprocessing import apply_benign_hiddens
from .utils.preprocessing import feature_selection
import core.utils.preprocessing

eps = np.finfo(float).eps


def warn(*args, **kwargs):
    pass


warnings.warn = warn


class TreeNode(object):
    def __init__(self):
        # Position info
        self.parent = None
        self.children = {}

        # Node info
        self.tag = 'ROOT'
        self.level = 0  # 0-based
        self.fold = 0
        self.children_tags = []
        self.children_number = 0

        # Classification info
        self.features_index = []
        self.train_index = []
        self.test_index = []
        self.test_index_all = []

        # Configuration info
        self.features_number = 0
        self.packets_number = 0
        self.classifier_name = 'srf'
        self.classifier_class = ''
        self.classifier_opts = ''
        self.detector_class = ''
        self.detector_opts = ''

        self.test_duration = 0.0
        self.complexity = 0.0

        self.label_encoder = None


class HierarchicalClassifier(object):

    def __init__(self, input_file, levels_number, max_level_target, level_target, features_number, packets_number,
                 classifier_class, classifier_opts, detector_class, detector_opts, workers_number, anomaly_classes,
                 epochs_number, arbitrary_discr, n_clusters, anomaly, deep, hidden_classes, benign_class,
                 nominal_features_index,
                 optimize, weight_features, parallelize, buckets_number, unsupervised):

        self.input_file = input_file
        self.levels_number = levels_number
        self.max_level_target = max_level_target
        self.level_target = level_target
        self.features_number = features_number
        self.packets_number = packets_number
        self.classifier_class = classifier_class
        self.classifier_opts = classifier_opts
        self.detector_class = detector_class
        self.detector_opts = detector_opts
        self.workers_number = workers_number
        self.has_config = False
        self.anomaly_classes = anomaly_classes
        self.epochs_number = epochs_number
        self.arbitrary_discr = arbitrary_discr
        self.n_clusters = n_clusters
        self.buckets_number = buckets_number

        self.anomaly = anomaly
        self.deep = deep
        self.hidden_classes = hidden_classes
        self.benign_class = benign_class

        self.nominal_features_index = nominal_features_index
        self.optimize = optimize
        self.weight_features = weight_features
        self.parallelize = parallelize
        self.unsupervised = unsupervised

        self.super_learner_default = ['ssv', 'sif']

        self.config_name = ''
        self.config = ''
        self.material_folder = ''
        self.material_proba_folder = ''
        self.material_features_folder = ''
        self.material_train_durations_folder = ''
        self.graph_folder = ''
        self.type_discr = ''
        self.params_discr = ''
        self.attributes_number = -1
        self.dataset_features_number = -1
        self.features = list()
        self.labels = list()
        self.features_names = list()
        self.numerical_features_index = list()
        self.fine_nominal_features_index = list()
        self.numerical_features_length = -1
        self.nominal_features_lengths = list()
        self.anomaly_class = ''

    def set_config(self, config_name, config):
        self.has_config = True
        self.config_name = config_name
        self.config = config

    def init_output_files(self):

        if self.anomaly:
            folder_discr = self.detector_class + '_'.join(self.detector_opts)
        else:
            folder_discr = self.classifier_class + '_'.join(self.classifier_opts)

        if self.has_config:
            folder_discr = self.config_name
        if len(self.anomaly_classes) > 0:
            folder_discr = self.detector_class + ''.join(['_' + str(opt) for opt in self.detector_opts])

        # File discriminator, modify params_discr to change dipendence variable
        self.type_discr = 'flow'
        feat_discr = '_f_' + str(self.features_number)
        if not self.has_config and self.packets_number != 0:
            self.type_discr = 'early'
            feat_discr = '_p_' + str(self.packets_number)
        elif self.has_config:
            if 'p' in self.config:
                self.type_discr = 'early'
            feat_discr = '_c_' + self.config_name
        if self.has_config and self.classifier_class:
            if self.features_number != 0:
                feat_discr = '_f_' + str(self.features_number) + feat_discr + '_' + self.classifier_class
            if self.packets_number != 0:
                feat_discr = '_p_' + str(self.packets_number) + feat_discr + '_' + self.classifier_class
        work_discr = '_w_' + str(self.workers_number)
        model_discr = '_e_' + str(self.epochs_number)
        buck_discr = '_b_' + str(self.buckets_number)

        self.params_discr = feat_discr
        if self.workers_number > 0:
            self.params_discr = work_discr + self.params_discr
            if self.buckets_number > 0:
                self.params_discr = buck_discr + self.params_discr
        elif self.epochs_number > 0:
            self.params_discr = model_discr + self.params_discr

        self.material_folder = './data_' + folder_discr + '/material/'
        self.material_proba_folder = './data_' + folder_discr + '/material_proba/'

        if not os.path.exists('./data_' + folder_discr):
            os.makedirs('./data_' + folder_discr)
            os.makedirs(self.material_folder)
            os.makedirs(self.material_proba_folder)
        else:
            if not os.path.exists(self.material_folder):
                os.makedirs(self.material_folder)
            if not os.path.exists(self.material_proba_folder):
                os.makedirs(self.material_proba_folder)

        self.material_features_folder = './data_' + folder_discr + '/material/features/'
        self.material_train_durations_folder = './data_' + folder_discr + '/material/train_durations/'

        if not os.path.exists(self.material_folder):
            os.makedirs(self.material_folder)
            os.makedirs(self.material_features_folder)
            os.makedirs(self.material_train_durations_folder)
        if not os.path.exists(self.material_features_folder):
            os.makedirs(self.material_features_folder)
        if not os.path.exists(self.material_train_durations_folder):
            os.makedirs(self.material_train_durations_folder)

        self.graph_folder = './data_' + folder_discr + '/graph/'

        if not os.path.exists('./data_' + folder_discr):
            os.makedirs('./data_' + folder_discr)
            os.makedirs(self.graph_folder)
        elif not os.path.exists(self.graph_folder):
            os.makedirs(self.graph_folder)

    def write_fold(self, level, tag, labels, _all=False, arbitrary_discr=None):

        if arbitrary_discr is None:
            arbitrary_discr = self.arbitrary_discr

        if _all:
            global_fouts = [
                '%s%s_multi_%s_level_%s%s_all.dat' % (
                    self.material_folder, arbitrary_discr, self.type_discr, level, self.params_discr),
                '%s%s_multi_%s_level_%s%s_all.dat' % (
                    self.material_proba_folder, arbitrary_discr, self.type_discr, level, self.params_discr)
            ]
            local_fouts = [
                '%s%s_multi_%s_level_%s%s_tag_%s_all.dat' % (
                    self.material_folder, arbitrary_discr, self.type_discr, level, self.params_discr, tag),
                '%s%s_multi_%s_level_%s%s_tag_%s_all.dat' % (
                    self.material_proba_folder, arbitrary_discr, self.type_discr, level, self.params_discr, tag)
            ]
            fouts = local_fouts
            if not self.global_folds_writed_all[level]:
                self.global_folds_writed_all[level] = True
                fouts += global_fouts
        else:
            global_fouts = [
                '%s%s_multi_%s_level_%s%s.dat' % (
                    self.material_folder, arbitrary_discr, self.type_discr, level, self.params_discr),
                '%s%s_multi_%s_level_%s%s.dat' % (
                    self.material_proba_folder, arbitrary_discr, self.type_discr, level, self.params_discr)
            ]
            local_fouts = [
                '%s%s_multi_%s_level_%s%s_tag_%s.dat' % (
                    self.material_folder, arbitrary_discr, self.type_discr, level, self.params_discr, tag),
                '%s%s_multi_%s_level_%s%s_tag_%s.dat' % (
                    self.material_proba_folder, arbitrary_discr, self.type_discr, level, self.params_discr, tag),
            ]
            fouts = local_fouts
            if not self.global_folds_writed[level]:
                self.global_folds_writed[level] = True
                fouts += global_fouts

        for fout in fouts:
            with open(fout, 'a') as file:
                if 'proba' in fout:
                    file.write('@fold %s\n' % ' '.join(['Actual'] + labels))
                else:
                    file.write('@fold Actual Pred\n')

    def write_pred(self, oracle, pred, level, tag, _all=False, arbitrary_discr=None):

        if arbitrary_discr is None:
            arbitrary_discr = self.arbitrary_discr

        if not _all:
            fouts = ['%s%s_multi_%s_level_%s%s.dat' % (
                self.material_folder, arbitrary_discr, self.type_discr, level, self.params_discr),
                     '%s%s_multi_%s_level_%s%s_tag_%s.dat' % (
                         self.material_folder, arbitrary_discr, self.type_discr, level, self.params_discr, tag)]
        else:
            fouts = ['%s%s_multi_%s_level_%s%s_all.dat' % (
                self.material_folder, arbitrary_discr, self.type_discr, level, self.params_discr),
                     '%s%s_multi_%s_level_%s%s_tag_%s_all.dat' % (
                         self.material_folder, arbitrary_discr, self.type_discr, level, self.params_discr, tag)]

        for fout in fouts:
            with open(fout, 'a') as file:
                for o, p in zip(oracle, pred):
                    file.write('%s %s\n' % (o, p))

    def write_bucket_times(self, bucket_times, arbitrary_discr=None):
        if arbitrary_discr is None:
            arbitrary_discr = self.arbitrary_discr

        with open(
                self.material_train_durations_folder + arbitrary_discr + 'multi_' + self.type_discr + self.params_discr + '_buckets_train_durations.dat',
                'a') as file0:
            file0.write('%.6f\n' % (np.max(bucket_times)))
            for i, bucket_time in enumerate(bucket_times):
                with open(
                        self.material_train_durations_folder + arbitrary_discr + 'multi_' + self.type_discr + self.params_discr + '_bucket_' + str(
                            i) + '_train_durations.dat', 'a') as file1:
                    file1.write('%.6f\n' % bucket_time)

    def write_time(self, time, level, tag, arbitrary_discr=None):
        if arbitrary_discr is None:
            arbitrary_discr = self.arbitrary_discr

        with open('%s%s_multi_%s_level_%s%s_tag_%s_train_durations.dat' % (
                self.material_train_durations_folder, arbitrary_discr, self.type_discr, level, self.params_discr, tag),
                  'a') as file:
            file.write('%.6f\n' % (np.max(time)))

    def write_proba(self, oracle, proba, level, tag, _all=False, arbitrary_discr=None):

        if arbitrary_discr is None:
            arbitrary_discr = self.arbitrary_discr

        if not _all:
            fouts = ['%s%s_multi_%s_level_%s%s.dat' % (
                self.material_proba_folder, arbitrary_discr, self.type_discr, level, self.params_discr),
                     '%s%s_multi_%s_level_%s%s_tag_%s.dat' % (
                         self.material_proba_folder, arbitrary_discr, self.type_discr, level, self.params_discr, tag)]
        else:
            fouts = ['%s%s_multi_%s_level_%s%s_all.dat' % (
                self.material_proba_folder, arbitrary_discr, self.type_discr, level, self.params_discr),
                     '%s%s_multi_%s_level_%s%s_tag_%s_all.dat' % (
                         self.material_proba_folder, arbitrary_discr, self.type_discr, level, self.params_discr, tag)]

        for fout in fouts:
            with open(fout, 'a') as file:
                for o, p in zip(oracle, proba):
                    try:
                        file.write('%s %s\n' % (o, ' '.join([str(v) for v in p])))
                    except:
                        file.write('%s %s\n' % (o, p))

    def load_image(self):

        with open(self.input_file, 'rb') as dataset_pk:
            dataset = pk.load(dataset_pk)

        data = np.array(dataset['data'], dtype=object)

        # After dataset preprocessing, we could save informations about dimensions
        self.attributes_number = data.shape[1]
        self.dataset_features_number = self.attributes_number - self.levels_number
        self.features = data[:, :self.dataset_features_number]
        self.labels = data[:, self.dataset_features_number:]

    def load_dataset(self, memoryless):
        '''
        :param memoryless: if True, the dataset wont be reloaded, else we use, if exists, the already processed version
        :return: no return, but instantiates some useful variables.
        '''
        if not memoryless and os.path.exists('%s.yetpreprocessed.pickle' % self.input_file):
            self.features_names, self.nominal_features_index, self.numerical_features_index, \
            self.fine_nominal_features_index, self.attributes_number, self.dataset_features_number, \
            self.features_number, self.anomaly_class, self.features, self.labels = pk.load(
                open('%s.yetpreprocessed.pickle' % self.input_file, 'rb'))
        else:
            # load .arff file
            with open(self.input_file, 'r') as fi:

                if self.input_file.lower().endswith('.arff'):
                    # load .arff file
                    dataset = arff.load(fi)
                elif self.input_file.lower().endswith('.csv'):
                    # TODO: to correct bugs
                    dataset = {}
                    delimiter = ','
                    attributes = fi.readline()
                    samples = fi.readlines()
                    first_row = samples[0].split(delimiter)
                    attrs_type = []
                    # Inferring attribute type by first row, this could led to error due to non analysis of the entire column
                    for value in first_row:
                        try:
                            float(value)
                            attrs_type.append(u'NUMERIC')
                        except:
                            attrs_type.append(u'STRING')
                    ##__##
                    dataset['attributes'] = [(attr, attr_type) for attr, attr_type in
                                             zip(attributes.strip().split(delimiter), attrs_type)]
                    dataset['data'] = [sample.strip().split(delimiter) for sample in
                                       samples]
                elif self.input_file.lower().endswith('.pickle'):
                    with open(self.input_file, 'rb') as dataset_pk:
                        dataset = pk.load(dataset_pk)

            data = np.array(dataset['data'], dtype=object)

            self.features_names = [x[0] for x in dataset['attributes']]

            # If is passed a detector that works with keras, perform a OneHotEncoding, else proceed with OrdinalEncoder
            if self.nominal_features_index is None:
                self.nominal_features_index = [i for i in range(len(dataset['attributes'][:-self.levels_number])) if
                                               dataset['attributes'][i][1] not in [u'NUMERIC', u'REAL', u'SPARRAY']]
            self.numerical_features_index = [i for i in range(len(dataset['attributes'][:-self.levels_number])) if
                                             dataset['attributes'][i][1] in [u'NUMERIC', u'REAL']]
            self.fine_nominal_features_index = [i for i in range(len(dataset['attributes'][:-self.levels_number])) if
                                                dataset['attributes'][i][1] in [u'SPARRAY']]

            # Preprocessing
            mode = core.utils.preprocessing.CUSTOM
            if not self.deep:
                mode += '-' + core.utils.preprocessing.FLATTEN
            data = core.utils.preprocessing.preprocess_dataset(data, self.nominal_features_index, mode)

            # Nominal features index should contains only string features that need to be processed with a one hot encoding.
            # These kind of features will be treated as sparse array, if deep learning models are used.
            # features_encoder = OneHotEncoder()
            # for i in self.nominal_features_index:
            #     data[:, i] = [sp for sp in features_encoder.fit_transform(data[:, i].reshape(-1, 1))]

            # If model is not deep, we need to flatten sparsed features, to permit an unbiased treatment of categorical featurees w.r.t numerical one.
            # if not self.deep:
            #     new_data = []
            #     for col in data.T:
            #         if issparse(col[0]):
            #             for col0 in np.array([c.todense() for c in col]).T:
            #                 new_data.append(col0[0])
            #         else:
            #             new_data.append(col)
            #     data = np.array(new_data).T

            # After dataset preprocessing, we could save informations about dimensions
            self.attributes_number = data.shape[1]
            self.dataset_features_number = self.attributes_number - self.levels_number

            # Moreover, we provide a count of nominal and numerical features, to correctly initialize models.
            # self.numerical_features_length = 0
            # self.nominal_features_lengths = []
            # for obj in data[0, :self.dataset_features_number]:
            #     if issparse(obj):
            #         self.nominal_features_lengths.append(obj.shape[1])
            #     else:
            #         self.numerical_features_length += 1
            # Bypassing the features selection
            # self.features_number = self.numerical_features_length + len(self.nominal_features_lengths)

            if self.anomaly and len(self.anomaly_classes) > 0:
                self.anomaly_class = apply_anomalies(data, self.dataset_features_number + self.level_target,
                                                     self.anomaly_classes)
            if self.benign_class != '':
                self.anomaly_class = apply_benign_hiddens(data, self.dataset_features_number + self.level_target,
                                                     self.benign_class, self.hidden_classes)

            self.features = data[:, :self.dataset_features_number]
            self.labels = data[:, self.dataset_features_number:]

            if not memoryless:
                pk.dump(
                    [
                        self.features_names, self.nominal_features_index, self.numerical_features_index,
                        self.fine_nominal_features_index, self.attributes_number, self.dataset_features_number,
                        self.features_number, self.anomaly_class, self.features, self.labels
                    ]
                    , open('%s.yetpreprocessed.pickle' % self.input_file, 'wb'), pk.HIGHEST_PROTOCOL)

    def load_early_dataset(self):

        # load .arff file
        if self.input_file.lower().endswith('.pickle'):
            with open(self.input_file, 'rb') as dataset_pk:
                dataset = pk.load(dataset_pk)
        else:
            raise (Exception('Early classification supports only pickle input.'))

        data = np.array(dataset['data'], dtype=object)

        # Next portion will merge all categorical features into one
        ###
        # nominal_features_index = [i for i in range(len(dataset['attributes'][:-self.levels_number])) if dataset['attributes'][i][1] != u'NUMERIC']

        # data_nominals = np.ndarray(shape=(self.data.shape[0],len(nominal_features_index)),dtype=object)
        # for i,nfi in enumerate(nominal_features_index):
        #     data_nominals[:,i] = self.data[:,nfi]
        # data_nominal = np.ndarray(shape=(self.data.shape[0]),dtype=object)
        # for i,datas in enumerate(data_nominals):
        #     data_nominal[i] = ''.join([ d for d in datas ])

        # self.data[:,nominal_features_index[0]] = data_nominal

        # self.data = np.delete(self.data, nominal_features_index[1:], axis=1).reshape((self.data.shape[0],self.data.shape[1]-len(nominal_features_index[1:])))
        ###

        self.features_names = [x[0] for x in dataset['attributes']]

        # If is passed a detector that works woth keras, perform a OneHotEncoding, else proceed with OrdinalEncoder
        if self.nominal_features_index is None:
            self.nominal_features_index = [i for i in range(len(dataset['attributes'][:-self.levels_number])) if
                                           dataset['attributes'][i][1] not in [u'NUMERIC', u'REAL', u'SPARRAY']]
        self.numerical_features_index = [i for i in range(len(dataset['attributes'][:-self.levels_number])) if
                                         dataset['attributes'][i][1] in [u'NUMERIC', u'REAL']]
        self.fine_nominal_features_index = [i for i in range(len(dataset['attributes'][:-self.levels_number])) if
                                            dataset['attributes'][i][1] in [u'SPARRAY']]

        # if self.deep:
        # if self.detector_class.startswith('k'):

        # Nominal features index should contains only string features that need to be processed with a one hot encoding.
        # These kind of features will be treated as sparse array, if deep learning models are used.
        features_encoder = OneHotEncoder()
        for i in self.nominal_features_index:
            # data[:, i] = [ sp.toarray()[0] for sp in features_encoder.fit_transform(data[:, i].reshape(-1,1)) ]
            features_encoder.fit(np.array([v for f in data[:, i] for v in f]).reshape(-1, 1))
            for j, f in enumerate(data[:, i]):
                data[:, i][j] = [sp for sp in features_encoder.transform(np.array(data[:, i][j]).reshape(-1, 1))]

        # After dataset preprocessing, we could save informations about dimensions
        self.attributes_number = data.shape[1]
        self.dataset_features_number = self.attributes_number - self.levels_number
        # Moreover, we provide a count of nominal and numerical features, to correctly initialize models.
        self.numerical_features_length = 0
        self.nominal_features_lengths = []
        for obj in data[0, :self.dataset_features_number]:
            if issparse(obj[0]):
                self.nominal_features_lengths.append(obj[0].shape[1])
            else:
                self.numerical_features_length += 1
        # Bypassing the features selection
        self.features_number = self.numerical_features_length + len(self.nominal_features_lengths)
        # else:
        # If classifier/detector is not deep, the framework categorize nominal features.
        # if len(self.nominal_features_index) > 0:
        #     features_encoder = OrdinalEncoder()
        #     data[:, self.nominal_features_index] = features_encoder.fit_transform(data[:, self.nominal_features_index])

        # self.distribution_features_index = [ i for i in range(len(dataset['attributes'][:-self.levels_number])) if dataset['attributes'][i][1] in [ u'SPARRAY' ] ]
        # for i in self.distribution_features_index:
        #     data[:, i] = [ sp.toarray()[0] for sp in data[:, i] ]

        # Multiclass AD - OneVSAll
        # TODO: now it works only on one level dataset for AD, where when multiclass and anomaly is specified, anomaly goes to '1' and others to '0'
        if self.anomaly and np.unique(data[:, self.attributes_number - 1]).shape[0] > 2:
            if len(self.anomaly_classes) > 0:
                for anomaly_class in self.anomaly_classes:
                    data[np.where(data[:,
                                  self.dataset_features_number + self.level_target] == anomaly_class), self.dataset_features_number + self.level_target] = '1'
                data[np.where(data[:,
                              self.dataset_features_number + self.level_target] != '1'), self.dataset_features_number + self.level_target] = '0'
                self.anomaly_class = '1'
        # TODO: manage hidden classes in presence of benign declared
        if self.benign_class != '':
            if len(self.hidden_classes) > 0:
                data[np.where((data[:, self.attributes_number - 1] != self.benign_class) & (data[:,
                                                                                            self.attributes_number - 1] != self.hidden_classes)), self.attributes_number - 1] = '1'
            else:
                data[np.where(
                    data[:, self.attributes_number - 1] != self.benign_class), self.attributes_number - 1] = '1'
            data[np.where(data[:, self.attributes_number - 1] == self.benign_class), self.attributes_number - 1] = '0'
            self.anomaly_class = '1'

        self.features = data[:, :self.dataset_features_number]
        self.labels = data[:, self.dataset_features_number:]

    def initialize_nodes(self, fold_cnt, train_index, test_index):

        nodes = []

        root = TreeNode()
        nodes.append(root)

        root.fold = fold_cnt
        root.level = self.level_target
        root.train_index = [index for index in train_index if self.labels[index, root.level] not in self.hidden_classes]
        root.test_index = test_index
        root.test_index_all = test_index
        root.children_tags = [tag for tag in np.unique((self.labels[:, root.level])) if tag not in self.hidden_classes]
        root.children_number = len(root.children_tags)
        root.label_encoder = MyLabelEncoder()
        root.label_encoder.fit(root.children_tags + self.hidden_classes)

        # Apply config to set number of features
        key = '%s_%s' % (root.tag, root.level + 1)
        if self.has_config and key in self.config:
            if 'f' in self.config[key]:
                root.features_number = self.config[key]['f']
            elif 'p' in self.config[key]:
                root.packets_number = self.config[key]['p']
            root.classifier_class = self.config[key]['c']
            classifier_to_call = getattr(self, supported_classifiers[root.classifier_class])

            # print('\nconfig', 'tag', root.tag, 'level', root.level, 'f', root.features_number,
            # 'c', root.classifier_class, 'train_test_len', len(root.train_index), len(root.test_index))
            print(
                'Initialize node tag %s at level %s with options: --features_number=%s --classifier_class=%s                            '
                % (root.tag, root.level + 1, root.features_number, root.classifier_class), end='\r')
        else:
            # Assign encoded features number to work with onehotencoder
            root.features_number = self.features_number
            # root.encoded_features_number = self.encoded_dataset_features_number
            root.packets_number = self.packets_number

            # If self.anomaly_class is set and it has two children (i.e. is a binary classification problem)
            # ROOT had to be an Anomaly Detector
            # In this experimentation we force the setting of AD to be a SklearnAnomalyDetector
            if self.anomaly and root.children_number == 2:
                root.detector_class = self.detector_class
                root.detector_opts = self.detector_opts
                # print('\nconfig', 'tag', root.tag, 'level', root.level, 'f', root.features_number,
                #       'd', root.detector_class, 'train_test_len', len(root.train_index), len(root.test_index))
                print(
                    'Initialize node tag %s at level %s with options: --features_number=%s --classifier_class=%s                            '
                    % (root.tag, root.level + 1, root.features_number, root.detector_class), end='\r')
                classifier_to_call = self._AnomalyDetector
            else:
                root.classifier_class = self.classifier_class
                root.classifier_opts = self.classifier_opts
                # print('\nconfig', 'tag', root.tag, 'level', root.level, 'f', root.features_number,
                #       'c', root.classifier_class, 'train_test_len', len(root.train_index), len(root.test_index))
                print(
                    'Initialize node tag %s at level %s with options: --features_number=%s --classifier_class=%s                            '
                    % (root.tag, root.level + 1, root.features_number, root.classifier_class), end='\r')
                classifier_to_call = getattr(self, supported_classifiers[root.classifier_class])

        root.classifier = classifier_to_call(node=root)

        # Creating hierarchy recursively, if max level target is set, classification continue while reaches it
        # If it's not set, it is equal to levels number. If level target is set, we have only the analysis at selected level.
        if root.level < self.max_level_target - 1 and root.children_number > 0:
            # print('\nInitializing nodes of Fold #%s\n' % fold_cnt)
            self.initialize_nodes_recursive(root, nodes)

        return nodes

    def initialize_nodes_recursive(self, parent, nodes):

        for i in range(parent.children_number):
            child = TreeNode()
            nodes.append(child)

            child.level = parent.level + 1
            child.fold = parent.fold
            child.tag = parent.children_tags[i]
            child.parent = parent

            child.train_index = [index for index in parent.train_index if self.labels[index, parent.level] == child.tag
                                 and self.labels[index, child.level] not in self.hidden_classes]
            child.test_index_all = [index for index in parent.test_index_all if
                                    self.labels[index, parent.level] == child.tag]
            child.children_tags = [tag for tag in np.unique((self.labels[child.train_index, child.level])) if
                                   tag not in self.hidden_classes]
            child.children_number = len(child.children_tags)
            # other_childs_children_tags correspond to the possible labels that in testing phase could arrive to the node
            other_childs_children_tags = [tag for tag in np.unique(self.labels[:, child.level]) if
                                          tag not in self.hidden_classes]
            child.label_encoder = MyLabelEncoder()
            child.label_encoder.fit(
                np.concatenate((child.children_tags, other_childs_children_tags, self.hidden_classes)))

            key = '%s_%s' % (child.tag, child.level + 1)
            if self.has_config and key in self.config:
                if 'f' in self.config[key]:
                    child.features_number = self.config[key]['f']
                elif 'p' in self.config[key]:
                    child.packets_number = self.config[key]['p']
                child.classifier_class = self.config[key]['c']
                # print('config', 'tag', child.tag, 'level', child.level, 'f', child.features_number,
                #       'c', child.classifier_class, 'd', child.detector_class, 'o', child.detector_opts,
                #       'train_test_len', len(child.train_index), len(child.test_index))
                print(
                    'Initialize node tag %s at level %s with options: --features_number=%s --classifier_class=%s                      '
                    % (child.tag, child.level + 1, child.features_number, child.classifier_class), end='\r')
            else:
                child.features_number = self.features_number
                # child.encoded_features_number = self.encoded_dataset_features_number
                child.packets_number = self.packets_number
                print(
                    'Initialize node tag %s at level %s with options: --features_number=%s --classifier_class=%s                      '
                    % (child.tag, child.level + 1, child.features_number, child.classifier_class), end='\r')

            if self.anomaly and child.children_number == 2:
                child.detector_class = self.detector_class
                child.detector_opts = self.detector_opts
            else:
                child.classifier_class = self.classifier_class
                child.classifier_opts = self.classifier_opts
            # print('config', 'tag', child.tag, 'level', child.level, 'f', child.features_number,
            #       'c', child.classifier_class, 'train_test_len', len(child.train_index), len(child.test_index))

            if self.anomaly and child.children_number == 2:
                classifier_to_call = self._AnomalyDetector
            else:
                classifier_to_call = getattr(self, supported_classifiers[child.classifier_class])
            child.classifier = classifier_to_call(node=child)

            nodes[nodes.index(parent)].children[child.tag] = child

            if child.level < self.max_level_target - 1 and child.children_number > 0:
                self.initialize_nodes_recursive(child, nodes)

    def kfold_validation(self, k=10, starting_fold=0, ending_fold=10):

        self.k = k
        classifiers_per_fold = []
        # bucket_times_per_fold = []

        skf = StratifiedKFold(n_splits=self.k, shuffle=True)

        # Testing set of each fold are not overlapping, we use only one array to maintain all the predictions over the folds
        self.predictions = np.ndarray(shape=self.labels.shape, dtype=object)

        # for fold_cnt, (train_index, test_index) in enumerate(skf.split(self.features, self.labels[:,-1])):
        #     with open('labels.dat','a') as fout:
        #         fout.write('@fold\n')
        #         for l in self.labels[test_index,1]:
        #             fout.write('%s\n'%l)

        for fold_cnt, (train_index, test_index) in enumerate(skf.split(self.features, self.labels[:, -1])):

            if fold_cnt < starting_fold or fold_cnt > ending_fold - 1:
                continue

            print('\nStarting Fold #%s\n' % (fold_cnt + 1))

            self.global_folds_writed = [False] * self.levels_number
            self.global_folds_writed_all = [False] * self.levels_number

            nodes = self.initialize_nodes(fold_cnt, train_index, test_index)

            print(
                'Nodes initialization DONE!                                                                                       ',
                end='\r')
            time.sleep(1)

            # Defining complexity for each node
            for node in [n for n in nodes if n.children_number > 1]:
                node.complexity = (len(node.train_index) * node.children_number) / np.sum(
                    [len(n.train_index) * n.children_number for n in nodes if n.children_number > 1])

            # Index of sorted (per complexity) nodes
            sorting_index = list(reversed(np.argsort([n.complexity for n in nodes if n.children_number > 1])))
            # Indexes of sorted nodes per bucket
            bucket_indexes = equal_partitioner(self.buckets_number,
                                               [[n for n in nodes if n.children_number > 1][i].complexity for i in
                                                sorting_index])

            if self.parallelize:
                # Iterative training, it is parallelizable leveraging bucketization:
                # buckets are launched in parallel, nodes in a bucket are trained sequentially
                def parallel_train(bucket_nodes):
                    for node in bucket_nodes:
                        print(
                            'Training node %s_%s                                                                                              '
                            % (node.tag, node.level + 1), end='\r')
                        self.train(node)

                threads = []
                for bucket_index in bucket_indexes:
                    bucket_nodes = [[n for n in nodes if n.children_number > 1][i] for i in
                                    [sorting_index[j] for j in bucket_index]]
                    threads.append(Thread(target=parallel_train, args=[bucket_nodes]))
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
            else:
                for bucket_index in bucket_indexes:
                    bucket_nodes = [[n for n in nodes if n.children_number > 1][i] for i in
                                    [sorting_index[j] for j in bucket_index]]
                    for node in bucket_nodes:
                        if node.children_number > 1:
                            print(
                                'Training node %s_%s                                                                                              '
                                % (node.tag, node.level + 1), end='\r')
                            self.train(node)

            print(
                'Nodes training DONE!                                                                                       ',
                end='\r')
            time.sleep(1)

            bucket_times = []
            for bucket_index in bucket_indexes:
                bucket_times.append(np.sum([nodes[i].test_duration for i in [sorting_index[j] for j in bucket_index]]))
            self.write_bucket_times(bucket_times)
            # bucket_times_per_fold.append(bucket_times)

            # Iterative testing_all, it is parallelizable in various ways
            for node in nodes:
                if node.children_number > 1:
                    print(
                        'Testing (all) node %s_%s                                                                                         '
                        % (node.tag, node.level + 1), end='\r')
                    self.write_fold(node.level, node.tag, node.children_tags, _all=True)
                    self.test_all(node)
            print(
                'Nodes testing (all) DONE!                                                                                       ',
                end='\r')
            time.sleep(1)

            # Recursive testing, each level is parallelizable and depends on previous level predictions
            root = nodes[0]
            print(
                'Testing node %s_%s                                                                                               '
                % (root.tag, root.level + 1), end='\r')
            self.write_fold(root.level, root.tag, root.children_tags)
            self.test(root)
            if root.level < self.max_level_target - 1 and root.children_number > 1:
                self.test_recursive(root)
            print(
                'Nodes testing DONE!                                                                                       ',
                end='\r')
            time.sleep(1)

            # Writing unary class predictions
            for node in nodes:
                if node.children_number == 1:
                    self.unary_class_results_inferring(node)
            classifiers_per_fold.append(nodes)

            print(
                'Fold #%s DONE!                                                                                            '
                % (fold_cnt + 1), end='\r')
            time.sleep(1)
            print('')

        # MEAN WEIGHT MODEL
        # if self.anomaly and self.deep:

        #     weights_per_fold = []
        #     # Weights information
        #     for classifiers in classifiers_per_fold:
        #         for node in classifiers:
        #             weights_per_fold.append(node.classifier.anomaly_detector.get_weights('r'))

        #     weights = np.mean(weights_per_fold, axis=0)

        #     for classifiers in classifiers_per_fold:
        #         for node in classifiers:
        #             node.classifier.anomaly_detector.set_weights(weights, 'r')
        #             # To distinguish among 10-fold realted and weighted ones
        #             self.params_discr = self.params_discr + '_MEAN'

        #     for classifiers in classifiers_per_fold:
        #         for node in classifiers:
        #             self.train(node)
        #             self.test(node)

        # for classifier in classifiers_per_fold[0]:

        #     with open(self.material_features_folder + self.arbitrary_discr + 'multi_' + self.type_discr + '_level_' + str(classifier.level+1) + self.params_discr + '_tag_' + str(classifier.tag) + '_features.dat', 'w+'):
        #         pass
        #     with open(self.material_train_durations_folder + self.arbitrary_discr + 'multi_' + self.type_discr + '_level_' + str(classifier.level+1) + self.params_discr + '_tag_' + str(classifier.tag) + '_train_durations.dat', 'w+'):
        #         pass

        # # for bucket_times in bucket_times_per_fold:
        # #     with open(self.material_train_durations_folder + self.arbitrary_discr + 'multi_' + self.type_discr + '_level_' + str(classifier.level+1) + self.params_discr + '_buckets_train_durations.dat', 'w+'):
        # #         pass
        # #     for i,_ in enumerate(bucket_times):
        # #         with open(self.material_train_durations_folder + self.arbitrary_discr + 'multi_' + self.type_discr + '_level_' + str(classifier.level+1) + self.params_discr + '_bucket_' + str(i) + '_train_durations.dat', 'w+'):
        # #             pass

        # with open(self.material_train_durations_folder + self.arbitrary_discr + 'multi_' + self.type_discr + self.params_discr + '_train_durations.dat', 'w+'):
        #     pass

        # for fold_n, classifiers in enumerate(classifiers_per_fold):

        #     for classifier in classifiers:

        #         # with open(self.material_features_folder + self.arbitrary_discr + 'multi_' + self.type_discr + '_level_' + str(classifier.level+1) + self.params_discr + '_tag_' + str(classifier.tag) + '_features.dat', 'a') as file:

        #         #     file.write('@fold\n')
        #         #     file.write(self.features_names[classifier.features_index[0]])

        #         #     for feature_index in classifier.features_index[1:]:
        #         #         file.write(','+self.features_names[feature_index])

        #         #     file.write('\n')

        #         with open(self.material_train_durations_folder + self.arbitrary_discr + 'multi_' + self.type_discr + '_level_' + str(classifier.level+1) + self.params_discr + '_tag_' + str(classifier.tag) + '_train_durations.dat', 'a') as file:

        #             file.write('%.6f\n' % (classifier.test_duration))

        # # Retrieve train_durations for each classifier
        # test_durations_per_fold = []

        # for classifiers in classifiers_per_fold:
        #     test_durations_per_fold.append([])
        #     for classifier in classifiers:
        #         test_durations_per_fold[-1].append(classifier.test_duration)

        # with open(self.material_train_durations_folder + self.arbitrary_discr + 'multi_' + self.type_discr + self.params_discr + '_train_durations.dat', 'w+') as file:

        #     mean_parallel_test_duration = np.mean(np.max(test_durations_per_fold, axis=1))
        #     std_parallel_test_duration = np.std(np.max(test_durations_per_fold, axis=1))

        #     mean_sequential_test_duration = np.mean(np.sum(test_durations_per_fold, axis=1))
        #     std_sequential_test_duration = np.std(np.sum(test_durations_per_fold, axis=1))

        #     file.write('mean_par,std_par,mean_seq,std_seq\n')
        #     file.write('%.6f,%.6f,%.6f,%.6f\n' % (mean_parallel_test_duration,std_parallel_test_duration,mean_sequential_test_duration,std_sequential_test_duration))

        # Graph plot
        G = nx.DiGraph()
        for info in classifiers_per_fold[0]:
            G.add_node(str(info.level) + ' ' + info.tag, level=info.level,
                       tag=info.tag, children_tags=info.children_tags)
        for node_parent, data_parent in G.nodes.items():
            for node_child, data_child in G.nodes.items():
                if data_child['level'] - data_parent['level'] == 1 and any(
                        data_child['tag'] in s for s in data_parent['children_tags']):
                    G.add_edge(node_parent, node_child)
        nx.write_gpickle(G,
                         self.graph_folder + self.arbitrary_discr + 'multi_' + self.type_discr + self.params_discr + '_graph.gml')

    def test_recursive(self, parent):

        for child_tag in parent.children:
            child = parent.children[child_tag]
            child.test_index = [index for index in parent.test_index if
                                self.predictions[index, parent.level] == child_tag]
            if child.children_number > 1:
                print(
                    'Testing node %s_%s                                                                                               '
                    % (child.tag, child.level + 1), end='\r')
                self.write_fold(child.level, child.tag, child.children_tags)
                self.test(child)
                if child.level < self.max_level_target - 1:
                    self.test_recursive(child)

    def unary_class_results_inferring(self, node):

        node.test_duration = 0.
        node.features_index = node.parent.features_index

        self.write_fold(node.level, node.tag, node.children_tags)
        self.write_fold(node.level, node.tag, node.children_tags, _all=True)
        self.write_pred(self.labels[node.test_index, node.level], [node.tag] * len(node.test_index), node.level,
                        node.tag)
        self.write_pred(self.labels[node.test_index_all, node.level], [node.tag] * len(node.test_index_all), node.level,
                        node.tag, True)

        if len(self.anomaly_classes) > 0:
            proba_base = (1., 1., 1.)
            for index in node.test_index:
                self.predictions[index, node.level] = node.tag
        #     self.probability[index, node.level] = (1., 1., 1.)
        # for index in node.test_index_all:
        #     self.prediction_all[index, node.level] = node.tag
        #     self.probability_all[index, node.level] = (1., 1., 1.)
        else:
            proba_base = 0.
            for index in node.test_index:
                self.predictions[index, node.level] = node.tag
        #     self.probability[index, node.level] = 0.
        # for index in node.test_index_all:
        #     self.prediction_all[index, node.level] = node.tag
        #     self.probability_all[index, node.level] = 0.

        self.write_proba(self.labels[node.test_index, node.level], [proba_base] * len(node.test_index), node.level,
                         node.tag)
        self.write_proba(self.labels[node.test_index_all, node.level], [proba_base] * len(node.test_index_all),
                         node.level, node.tag, True)

    # TODO: does not apply preprocessing to dataset
    def Sklearn_RandomForest(self, node):
        # Instantation
        classifier = OutputCodeClassifier(RandomForestClassifier(n_estimators=100, n_jobs=-1),
                                          code_size=np.ceil(np.log2(node.children_number) / node.children_number))
        # # Features selection
        # node.features_index = self.features_selection(node)
        return classifier

    # TODO: does not apply preprocessing to dataset
    def Sklearn_CART(self, node):
        # Instantation
        classifier = DecisionTreeClassifier()
        # # Features selection
        # node.features_index = self.features_selection(node)
        return classifier

    def Keras_Classifier(self, node):
        # Instantation
        classifier = SklearnKerasWrapper(*node.classifier_opts, model_class=node.classifier_class,
                                         epochs_number=self.epochs_number,
                                         num_classes=node.children_number,
                                         nominal_features_index=self.nominal_features_index,
                                         fine_nominal_features_index=self.fine_nominal_features_index,
                                         numerical_features_index=self.numerical_features_index, level=node.level,
                                         fold=node.fold, classify=True,
                                         weight_features=self.weight_features, arbitrary_discr=self.arbitrary_discr)
        # # Features selection
        # node.features_index = self.features_selection(node)
        return classifier

    def Keras_StackedDeepAutoencoderClassifier(self, node):
        return self.Keras_Classifier(node)

    def Keras_MultiLayerPerceptronClassifier(self, node):
        return self.Keras_Classifier(node)

    def Keras_Convolutional2DAutoencoder(self, node):
        return self.Keras_Classifier(node)

    def _AnomalyDetector(self, node):
        # Instantation
        classifier = AnomalyDetector(node.detector_class, node.detector_opts,
                                     node.label_encoder.transform([self.anomaly_class])[0], node.features_number,
                                     self.epochs_number, node.level, node.fold, self.n_clusters, self.optimize,
                                     self.weight_features, self.workers_number, self.unsupervised)
        # # Features selection
        # node.features_index = self.feature_selection(node)
        return classifier

    def Spark_Classifier(self, node):
        # Instantation
        classifier = SklearnSparkWrapper(classifier_class=node.classifier_class,
                                         num_classes=node.children_number,
                                         numerical_features_index=self.numerical_features_index,
                                         nominal_features_index=self.nominal_features_index,
                                         fine_nominal_features_index=self.fine_nominal_features_index,
                                         classifier_opts=node.classifier_opts,
                                         epochs_number=self.epochs_number, level=node.level, fold=node.fold,
                                         classify=True, workers_number=self.workers_number,
                                         arbitrary_discr=self.arbitrary_discr)
        # # Features selection
        # node.features_index = self.feature_selection(node)
        classifier.set_oracle(
            node.label_encoder.transform(self.labels[node.test_index, node.level])
        )
        return classifier

    def Spark_RandomForest(self, node):
        return self.Spark_Classifier(node)

    def Spark_NaiveBayes(self, node):
        return self.Spark_Classifier(node)

    def Spark_Keras_StackedDeepAutoencoderClassifier(self, node):
        return self.Spark_Classifier(node)

    def Spark_Keras_MultiLayerPerceptronClassifier(self, node):
        return self.Spark_Classifier(node)

    def Spark_MultilayerPerceptron(self, node):
        return self.Spark_Classifier(node)

    def Spark_GBT(self, node):
        return self.Spark_Classifier(node)

    def Spark_DecisionTree(self, node):
        return self.Spark_Classifier(node)

    def Weka_Classifier(self, node):
        # Instantation
        classifier = SklearnWekaWrapper(node.classifier_class
                                        # , self.nominal_features_index
                                        )
        # # Features selection
        # node.features_index = self.feature_selection(node)
        return classifier

    def Weka_NaiveBayes(self, node):
        return self.Weka_Classifier(node)

    def Weka_BayesNetwork(self, node):
        return self.Weka_Classifier(node)

    def Weka_RandomForest(self, node):
        return self.Weka_Classifier(node)

    def Weka_J48(self, node):
        return self.Weka_Classifier(node)

    def Weka_SuperLearner(self, node):
        # Instantation
        # classifier = SuperLearnerClassifier(['ssv', 'sif', 'slo', 'slo1', 'slo2', 'ssv1', 'ssv2', 'ssv3', 'ssv4'], node.children_number, node.label_encoder.transform(self.anomaly_class)[0])
        classifier = SuperLearnerClassifier(self.super_learner_default, node.children_number,
                                            node.label_encoder.transform([self.anomaly_class])[0], node.features_number)
        # # Features selection
        # node.features_index = self.feature_selection(node)
        classifier.set_oracle(
            node.label_encoder.transform(self.labels[node.test_index_all, node.level])
        )
        return classifier

    def train(self, node):
        node.features_index = feature_selection(
            self.features[node.train_index],
            self.labels[node.train_index, node.level],
            node,
            self.dataset_features_number
        )
        node.test_duration = node.classifier.fit(
            self.features[node.train_index][:, node.features_index],
            node.label_encoder.transform(self.labels[node.train_index, node.level])
        ).t_
        print(
            'Training %s_%s duration [s]: %s                                                                                  '
            % (node.tag, node.level + 1, node.test_duration), end='\r')
        time.sleep(1)

        self.write_time(node.test_duration, node.level, node.tag)

    def test(self, node):
        if len(node.test_index) == 0:
            return
        node.classifier.set_oracle(
            node.label_encoder.transform(self.labels[node.test_index, node.level])
        )
        pred = node.label_encoder.inverse_transform(
            node.classifier.predict(self.features[node.test_index][:, node.features_index]))
        try:
            proba = node.classifier.predict_proba(self.features[node.test_index][:, node.features_index])
        except:
            proba = node.classifier.score(self.features[node.test_index][:, node.features_index])

        for p, b, i in zip(pred, proba, node.test_index):
            self.predictions[i, node.level] = p

        self.write_pred(self.labels[node.test_index, node.level], pred, node.level, node.tag)
        self.write_proba(self.labels[node.test_index, node.level], proba, node.level, node.tag)

    def test_all(self, node):
        node.classifier.set_oracle(
            node.label_encoder.transform(self.labels[node.test_index_all, node.level])
        )
        pred = node.label_encoder.inverse_transform(
            node.classifier.predict(self.features[node.test_index_all][:, node.features_index]))
        try:
            proba = node.classifier.predict_proba(self.features[node.test_index_all][:, node.features_index])
        except:
            proba = node.classifier.score(self.features[node.test_index_all][:, node.features_index])

        # for p, b, i in zip(pred, proba, node.test_index_all):
        #     self.prediction_all[i, node.level] = p
        #     self.probability_all[i, node.level] = b

        self.write_pred(self.labels[node.test_index_all, node.level], pred, node.level, node.tag, True)
        self.write_proba(self.labels[node.test_index_all, node.level], proba, node.level, node.tag, True)
