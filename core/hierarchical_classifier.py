#!/usr/bin/python3

import os
import pickle as pk
from threading import Thread
import time

import arff
import networkx as nx
import numpy as np

import warnings

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
from core.wrappers.weka_wrapper import serialization

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
        self.model_wrapper = None
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

        # Complexity info
        self.training_duration = 0.0
        self.testing_duration = 0.0
        self.testing_all_duration = 0.0
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
        self.material_preds_folder = ''
        self.material_rawpreds_folder = ''
        self.material_features_folder = ''
        self.material_durations_folder = ''
        self.material_graphs_folder = ''
        self.material_models_folder = ''
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
            folder_discr = self.detector_class
            if len(self.detector_opts) > 0:
            	folder_discr += '_' + '_'.join(self.detector_opts)
        else:
            folder_discr = self.classifier_class
            if len(self.classifier_opts) > 0:
            	folder_discr += '_' + '_'.join(self.classifier_opts)

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

        material_folder = './data_%s/material/' % folder_discr

        self.material_preds_folder = '%s/preds/' % material_folder
        self.material_rawpreds_folder = '%s/rawpreds/' % material_folder

        if not os.path.exists('./data_%s/' % folder_discr):
            os.makedirs('./data_%s/' % folder_discr)
            os.makedirs(material_folder)
            os.makedirs(self.material_preds_folder)
            os.makedirs(self.material_rawpreds_folder)
        else:
            if not os.path.exists(self.material_preds_folder):
                os.makedirs(self.material_preds_folder)
            if not os.path.exists(self.material_rawpreds_folder):
                os.makedirs(self.material_rawpreds_folder)

        self.material_features_folder = '%s/features/' % material_folder
        self.material_durations_folder = '%s/durations/' % material_folder

        if not os.path.exists(self.material_features_folder):
            os.makedirs(self.material_features_folder)
        if not os.path.exists(self.material_durations_folder):
            os.makedirs(self.material_durations_folder)

        self.material_graphs_folder = '%s/graphs/' % material_folder

        if not os.path.exists(self.material_graphs_folder):
            os.makedirs(self.material_graphs_folder)

        self.material_models_folder = '%s/models/' % material_folder

        if not os.path.exists(self.material_models_folder):
            os.makedirs(self.material_models_folder)

    def write_fold(self, level, tag, labels, _all=False, arbitrary_discr=None):

        if arbitrary_discr is None:
            arbitrary_discr = self.arbitrary_discr

        if _all:
            global_fouts = [
                '%s%s_multi_%s_level_%s%s_all.dat' % (
                    self.material_preds_folder, arbitrary_discr, self.type_discr, level, self.params_discr),
                '%s%s_multi_%s_level_%s%s_all.dat' % (
                    self.material_rawpreds_folder, arbitrary_discr, self.type_discr, level, self.params_discr)
            ]
            local_fouts = [
                '%s%s_multi_%s_level_%s%s_tag_%s_all.dat' % (
                    self.material_preds_folder, arbitrary_discr, self.type_discr, level, self.params_discr, tag),
                '%s%s_multi_%s_level_%s%s_tag_%s_all.dat' % (
                    self.material_rawpreds_folder, arbitrary_discr, self.type_discr, level, self.params_discr, tag)
            ]
            fouts = local_fouts
            if not self.global_folds_writed_all[level]:
                self.global_folds_writed_all[level] = True
                fouts += global_fouts
        else:
            global_fouts = [
                '%s%s_multi_%s_level_%s%s.dat' % (
                    self.material_preds_folder, arbitrary_discr, self.type_discr, level, self.params_discr),
                '%s%s_multi_%s_level_%s%s.dat' % (
                    self.material_rawpreds_folder, arbitrary_discr, self.type_discr, level, self.params_discr)
            ]
            local_fouts = [
                '%s%s_multi_%s_level_%s%s_tag_%s.dat' % (
                    self.material_preds_folder, arbitrary_discr, self.type_discr, level, self.params_discr, tag),
                '%s%s_multi_%s_level_%s%s_tag_%s.dat' % (
                    self.material_rawpreds_folder, arbitrary_discr, self.type_discr, level, self.params_discr, tag),
            ]
            fouts = local_fouts
            if not self.global_folds_writed[level]:
                self.global_folds_writed[level] = True
                fouts += global_fouts

        for fout in fouts:
            with open(fout, 'a') as file:
                if self.anomaly:
                	file.write('@fold Actual Pred\n')
                else:
                	file.write('@fold %s\n' % ' '.join(['Actual'] + labels))

    def write_pred(self, oracle, pred, level, tag, _all=False, arbitrary_discr=None):

        if arbitrary_discr is None:
            arbitrary_discr = self.arbitrary_discr

        if not _all:
            fouts = ['%s%s_multi_%s_level_%s%s.dat' % (
                self.material_preds_folder, arbitrary_discr, self.type_discr, level, self.params_discr),
                     '%s%s_multi_%s_level_%s%s_tag_%s.dat' % (
                         self.material_preds_folder, arbitrary_discr, self.type_discr, level, self.params_discr, tag)]
        else:
            fouts = ['%s%s_multi_%s_level_%s%s_all.dat' % (
                self.material_preds_folder, arbitrary_discr, self.type_discr, level, self.params_discr),
                     '%s%s_multi_%s_level_%s%s_tag_%s_all.dat' % (
                         self.material_preds_folder, arbitrary_discr, self.type_discr, level, self.params_discr, tag)]

        for fout in fouts:
            with open(fout, 'a') as file:
                for o, p in zip(oracle, pred):
                    file.write('%s %s\n' % (o, p))

    def write_bucket_times(self, bucket_times, arbitrary_discr=None):
        if arbitrary_discr is None:
            arbitrary_discr = self.arbitrary_discr
        else:
            arbitrary_discr = '%s_%s_' % (self.arbitrary_discr, arbitrary_discr)

        with open('%s%s_multi_%s_%s_buckets_train_durations.dat' % (
                self.material_durations_folder, arbitrary_discr, self.type_discr, self.params_discr),
                  'a') as file0:
            file0.write('%.6f\n' % (np.max(bucket_times)))
            for i, bucket_time in enumerate(bucket_times):
                with open('%s%s_multi_%s%s_bucket_%s_train_durations.dat' % (
                        self.material_durations_folder, arbitrary_discr, self.type_discr, self.params_discr, i)
                        , 'a') as file1:
                    file1.write('%.6f\n' % bucket_time)

    def write_time(self, time, level, tag, arbitrary_discr=None):
        if arbitrary_discr is None:
            arbitrary_discr = self.arbitrary_discr
        else:
            arbitrary_discr = '%s_%s_' % (self.arbitrary_discr, arbitrary_discr)

        with open('%s%s_multi_%s_level_%s%s_tag_%s_train_durations.dat' % (
                self.material_durations_folder, arbitrary_discr, self.type_discr, level, self.params_discr, tag),
                  'a') as file:
            file.write('%.6f\n' % (np.max(time)))

    def write_proba(self, oracle, proba, level, tag, _all=False, arbitrary_discr=None):

        if arbitrary_discr is None:
            arbitrary_discr = self.arbitrary_discr

        if not _all:
            fouts = ['%s%s_multi_%s_level_%s%s.dat' % (
                self.material_rawpreds_folder, arbitrary_discr, self.type_discr, level, self.params_discr),
                     '%s%s_multi_%s_level_%s%s_tag_%s.dat' % (
                         self.material_rawpreds_folder, arbitrary_discr, self.type_discr, level, self.params_discr,
                         tag)]
        else:
            fouts = ['%s%s_multi_%s_level_%s%s_all.dat' % (
                self.material_rawpreds_folder, arbitrary_discr, self.type_discr, level, self.params_discr),
                     '%s%s_multi_%s_level_%s%s_tag_%s_all.dat' % (
                         self.material_rawpreds_folder, arbitrary_discr, self.type_discr, level, self.params_discr,
                         tag)]

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

            # Pre-assigment of temporary values
            self.attributes_number = data.shape[1]
            self.dataset_features_number = self.attributes_number - self.levels_number

            self.features = data[:, :self.dataset_features_number]
            self.labels = data[:, self.dataset_features_number:]

            # Preprocessing
            if self.detector_class == 'mlo':
                mode = core.utils.preprocessing.MLO
            else:
                mode = core.utils.preprocessing.CUSTOM
                if not self.deep:
                    mode += '-' + core.utils.preprocessing.FLATTEN
            self.features = core.utils.preprocessing.preprocess_dataset(self.features, self.nominal_features_index,
                                                                        mode)

            # After dataset preprocessing, we could update informations about dimensions
            self.attributes_number = self.features.shape[1] + self.levels_number
            self.dataset_features_number = self.features.shape[1]

            if self.anomaly and len(self.anomaly_classes) > 0:
                self.anomaly_class = apply_anomalies(self.labels, self.level_target, self.anomaly_classes)
            if self.benign_class != '':
                self.anomaly_class = apply_benign_hiddens(self.labels, self.level_target, self.benign_class,
                                                          self.hidden_classes)

            if not memoryless:
                pk.dump(
                    [
                        self.features_names, self.nominal_features_index, self.numerical_features_index,
                        self.fine_nominal_features_index, self.attributes_number, self.dataset_features_number,
                        self.features_number, self.anomaly_class, self.features, self.labels
                    ]
                    , open('%s.yetpreprocessed.pickle' % self.input_file, 'wb'), pk.HIGHEST_PROTOCOL)

    def load_early_dataset(self):
        pass

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
            model_to_call = getattr(self, supported_classifiers[root.classifier_class])

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
                print(
                    'Initialize node tag %s at level %s with options: --features_number=%s --classifier_class=%s                            '
                    % (root.tag, root.level + 1, root.features_number, root.detector_class), end='\r')
                model_to_call = self._AnomalyDetector
            else:
                root.classifier_class = self.classifier_class
                root.classifier_opts = self.classifier_opts
                print(
                    'Initialize node tag %s at level %s with options: --features_number=%s --classifier_class=%s                            '
                    % (root.tag, root.level + 1, root.features_number, root.classifier_class), end='\r')
                model_to_call = getattr(self, supported_classifiers[root.classifier_class])

        root.model_wrapper = model_to_call(node=root)

        # Creating hierarchy recursively, if max level target is set, classification continue while reaches it
        # If it's not set, it is equal to levels number. If level target is set, we have only the analysis at selected level.
        if root.level < self.max_level_target - 1 and root.children_number > 0:
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
                print(
                    'Initialize node tag %s at level %s with options: --features_number=%s --classifier_class=%s                      '
                    % (child.tag, child.level + 1, child.features_number, child.classifier_class), end='\r')
            else:
                child.features_number = self.features_number
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

            if self.anomaly and child.children_number == 2:
                model_to_call = self._AnomalyDetector
            else:
                model_to_call = getattr(self, supported_classifiers[child.classifier_class])
            child.model_wrapper = model_to_call(node=child)

            nodes[nodes.index(parent)].children[child.tag] = child

            if child.level < self.max_level_target - 1 and child.children_number > 0:
                self.initialize_nodes_recursive(child, nodes)

    def kfold_validation(self, k=10, starting_fold=0, ending_fold=10):

        self.k = k
        models_per_fold = []

        skf = StratifiedKFold(n_splits=self.k, shuffle=True)

        # Testing set of each fold are not overlapping, we use only one array to maintain all the predictions over the folds
        self.predictions = np.ndarray(shape=self.labels.shape, dtype=object)

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

            bucket_training_times = []
            bucket_testing_times = []
            for bucket_index in bucket_indexes:
                bucket_training_times.append(
                    np.sum([nodes[i].training_duration for i in [sorting_index[j] for j in bucket_index]]))
                bucket_testing_times.append(
                    np.sum([nodes[i].testing_duration for i in [sorting_index[j] for j in bucket_index]]))
            self.write_bucket_times(bucket_training_times, arbitrary_discr='training')
            self.write_bucket_times(bucket_testing_times, arbitrary_discr='testing')

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
            models_per_fold.append(nodes)

            print(
                'Fold #%s DONE!                                                                                            '
                % (fold_cnt + 1), end='\r')
            time.sleep(1)
            print('')

            for node in nodes:
                try:
                    pk.dump(node.model_wrapper.model_, open('%s%s_multi_%s_fold_%s_level_%s%s_tag_%s_model.pickle' %
                            (self.material_models_folder, self.arbitrary_discr, self.type_discr, fold_cnt, node.level, self.params_discr, node.tag),'wb'))
                except:
                	serialization.write('%s%s_multi_%s_fold_%s_level_%s%s_tag_%s_model.model' %
                            (self.material_models_folder, self.arbitrary_discr, self.type_discr, fold_cnt, node.level, self.params_discr, node.tag),
                            node.model_wrapper.model_)
        # MEAN WEIGHT MODEL
        # if self.anomaly and self.deep:

        #     weights_per_fold = []
        #     # Weights information
        #     for models in models_per_fold:
        #         for node in models:
        #             weights_per_fold.append(node.model_wrapper.anomaly_detector.get_weights('r'))

        #     weights = np.mean(weights_per_fold, axis=0)

        #     for models in models_per_fold:
        #         for node in models:
        #             node.model_wrapper.anomaly_detector.set_weights(weights, 'r')
        #             # To distinguish among 10-fold realted and weighted ones
        #             self.params_discr = self.params_discr + '_MEAN'

        #     for models in models_per_fold:
        #         for node in models:
        #             self.train(node)
        #             self.test(node)

        # for model in models_per_fold[0]:

        #     with open(self.material_features_folder + self.arbitrary_discr + 'multi_' + self.type_discr + '_level_' + str(model.level+1) + self.params_discr + '_tag_' + str(model.tag) + '_features.dat', 'w+'):
        #         pass
        #     with open(self.material_durations_folder + self.arbitrary_discr + 'multi_' + self.type_discr + '_level_' + str(model.level+1) + self.params_discr + '_tag_' + str(model.tag) + '_train_durations.dat', 'w+'):
        #         pass

        # # for bucket_times in bucket_times_per_fold:
        # #     with open(self.material_durations_folder + self.arbitrary_discr + 'multi_' + self.type_discr + '_level_' + str(model.level+1) + self.params_discr + '_buckets_train_durations.dat', 'w+'):
        # #         pass
        # #     for i,_ in enumerate(bucket_times):
        # #         with open(self.material_durations_folder + self.arbitrary_discr + 'multi_' + self.type_discr + '_level_' + str(model.level+1) + self.params_discr + '_bucket_' + str(i) + '_train_durations.dat', 'w+'):
        # #             pass

        # with open(self.material_durations_folder + self.arbitrary_discr + 'multi_' + self.type_discr + self.params_discr + '_train_durations.dat', 'w+'):
        #     pass

        # for fold_n, models in enumerate(models_per_fold):

        #     for model in models:

        #         # with open(self.material_features_folder + self.arbitrary_discr + 'multi_' + self.type_discr + '_level_' + str(model.level+1) + self.params_discr + '_tag_' + str(model.tag) + '_features.dat', 'a') as file:

        #         #     file.write('@fold\n')
        #         #     file.write(self.features_names[model.features_index[0]])

        #         #     for feature_index in model.features_index[1:]:
        #         #         file.write(','+self.features_names[feature_index])

        #         #     file.write('\n')

        #         with open(self.material_durations_folder + self.arbitrary_discr + 'multi_' + self.type_discr + '_level_' + str(model.level+1) + self.params_discr + '_tag_' + str(model.tag) + '_train_durations.dat', 'a') as file:

        #             file.write('%.6f\n' % (model.test_duration))

        # # Retrieve train_durations for each model
        # test_durations_per_fold = []

        # for models in models_per_fold:
        #     test_durations_per_fold.append([])
        #     for model in models:
        #         test_durations_per_fold[-1].append(model.test_duration)

        # with open(self.material_durations_folder + self.arbitrary_discr + 'multi_' + self.type_discr + self.params_discr + '_train_durations.dat', 'w+') as file:

        #     mean_parallel_test_duration = np.mean(np.max(test_durations_per_fold, axis=1))
        #     std_parallel_test_duration = np.std(np.max(test_durations_per_fold, axis=1))

        #     mean_sequential_test_duration = np.mean(np.sum(test_durations_per_fold, axis=1))
        #     std_sequential_test_duration = np.std(np.sum(test_durations_per_fold, axis=1))

        #     file.write('mean_par,std_par,mean_seq,std_seq\n')
        #     file.write('%.6f,%.6f,%.6f,%.6f\n' % (mean_parallel_test_duration,std_parallel_test_duration,mean_sequential_test_duration,std_sequential_test_duration))

        # Graph plot
        G = nx.DiGraph()
        for info in models_per_fold[0]:
            G.add_node(str(info.level) + ' ' + info.tag, level=info.level,
                       tag=info.tag, children_tags=info.children_tags)
        for node_parent, data_parent in G.nodes.items():
            for node_child, data_child in G.nodes.items():
                if data_child['level'] - data_parent['level'] == 1 and any(
                        data_child['tag'] in s for s in data_parent['children_tags']):
                    G.add_edge(node_parent, node_child)
        nx.write_gpickle(G,
                         self.material_graphs_folder + self.arbitrary_discr + 'multi_' + self.type_discr + self.params_discr + '_graph.gml')

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
        else:
            proba_base = 0.
            for index in node.test_index:
                self.predictions[index, node.level] = node.tag

        self.write_proba(self.labels[node.test_index, node.level], [proba_base] * len(node.test_index), node.level,
                         node.tag)
        self.write_proba(self.labels[node.test_index_all, node.level], [proba_base] * len(node.test_index_all),
                         node.level, node.tag, True)

    # TODO: does not apply preprocessing to dataset (if applied per-node)
    def Sklearn_RandomForest(self, node):
        # Instantation
        model = OutputCodeClassifier(RandomForestClassifier(n_estimators=100, n_jobs=-1),
                                          code_size=np.ceil(np.log2(node.children_number) / node.children_number))
        return model

    # TODO: does not apply preprocessing to dataset (if applied per-node)
    def Sklearn_CART(self, node):
        # Instantation
        model = DecisionTreeClassifier()
        return model

    def Keras_Classifier(self, node):
        # Instantation
        model = SklearnKerasWrapper(*node.classifier_opts, model_class=node.classifier_class,
                                         epochs_number=self.epochs_number,
                                         num_classes=node.children_number,
                                         nominal_features_index=self.nominal_features_index,
                                         fine_nominal_features_index=self.fine_nominal_features_index,
                                         numerical_features_index=self.numerical_features_index, level=node.level,
                                         fold=node.fold, classify=True,
                                         weight_features=self.weight_features, arbitrary_discr=self.arbitrary_discr)
        return model

    def Keras_StackedDeepAutoencoderClassifier(self, node):
        return self.Keras_Classifier(node)

    def Keras_MultiLayerPerceptronClassifier(self, node):
        return self.Keras_Classifier(node)

    def Keras_Convolutional2DAutoencoder(self, node):
        return self.Keras_Classifier(node)

    def _AnomalyDetector(self, node):
        # Instantation
        model = AnomalyDetector(node.detector_class, node.detector_opts,
                                     node.label_encoder.transform([self.anomaly_class])[0], node.features_number,
                                     self.epochs_number, node.level, node.fold, self.n_clusters, self.optimize,
                                     self.weight_features, self.workers_number, self.unsupervised, self.arbitrary_discr)
        return model

    def Spark_Classifier(self, node):
        # Instantation
        model = SklearnSparkWrapper(classifier_class=node.classifier_class,
                                         num_classes=node.children_number,
                                         numerical_features_index=self.numerical_features_index,
                                         nominal_features_index=self.nominal_features_index,
                                         fine_nominal_features_index=self.fine_nominal_features_index,
                                         classifier_opts=node.classifier_opts,
                                         epochs_number=self.epochs_number, level=node.level, fold=node.fold,
                                         classify=True, workers_number=self.workers_number,
                                         arbitrary_discr=self.arbitrary_discr)
        model.set_oracle(
            node.label_encoder.transform(self.labels[node.test_index, node.level])
        )
        return model

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
        model = SklearnWekaWrapper(node.classifier_class)
        return model

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
        model = SuperLearnerClassifier(self.super_learner_default, node.children_number,
                                            node.label_encoder.transform([self.anomaly_class])[0], node.features_number)
        model.set_oracle(
            node.label_encoder.transform(self.labels[node.test_index_all, node.level])
        )
        return model

    def train(self, node):
        # TODO: applying preprocessing per node
        node.features_index = feature_selection(
            self.features[node.train_index],
            node.label_encoder.transform(self.labels[node.train_index, node.level]),
            node.features_number,
            self.dataset_features_number
        )
        node.model_wrapper.fit(
            self.features[node.train_index][:, node.features_index],
            node.label_encoder.transform(self.labels[node.train_index, node.level])
        )
        node.training_duration = node.model_wrapper.tr_
        print(
            'Training %s_%s duration [s]: %s                                                                                  '
            % (node.tag, node.level + 1, node.training_duration), end='\r')
        time.sleep(1)

        self.write_time(node.training_duration, node.level, node.tag, arbitrary_discr='training')

    def test(self, node):
        if len(node.test_index) == 0:
            return
        node.model_wrapper.set_oracle(
            node.label_encoder.transform(self.labels[node.test_index, node.level])
        )
        pred = node.label_encoder.inverse_transform(
            node.model_wrapper.predict(self.features[node.test_index][:, node.features_index]))
        try:
            proba = node.model_wrapper.predict_proba(self.features[node.test_index][:, node.features_index])
        except:
            proba = node.model_wrapper.score(self.features[node.test_index][:, node.features_index])
        for p, b, i in zip(pred, proba, node.test_index):
            self.predictions[i, node.level] = p
        node.testing_duration = node.model_wrapper.te_
        print(
            'Testing %s_%s duration [s]: %s                                                                                  '
            % (node.tag, node.level + 1, node.testing_duration), end='\r')
        time.sleep(1)

        self.write_pred(self.labels[node.test_index, node.level], pred, node.level, node.tag)
        self.write_proba(self.labels[node.test_index, node.level], proba, node.level, node.tag)
        self.write_time(node.testing_duration, node.level, node.tag, arbitrary_discr='testing')

    def test_all(self, node):
        node.model_wrapper.set_oracle(
            node.label_encoder.transform(self.labels[node.test_index_all, node.level])
        )
        pred = node.label_encoder.inverse_transform(
            node.model_wrapper.predict(self.features[node.test_index_all][:, node.features_index]))
        try:
            proba = node.model_wrapper.predict_proba(self.features[node.test_index_all][:, node.features_index])
        except:
            proba = node.model_wrapper.score(self.features[node.test_index_all][:, node.features_index])

        node.testing_all_duration = node.model_wrapper.te_
        print(
            'Testing (all) %s_%s duration [s]: %s                                                                                  '
            % (node.tag, node.level + 1, node.testing_all_duration), end='\r')
        time.sleep(1)

        self.write_pred(self.labels[node.test_index_all, node.level], pred, node.level, node.tag, True)
        self.write_proba(self.labels[node.test_index_all, node.level], proba, node.level, node.tag, True)
        self.write_time(node.testing_all_duration, node.level, node.tag, arbitrary_discr='testing_all')
