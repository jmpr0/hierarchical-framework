import numpy as np
import time
from sklearn.preprocessing import OrdinalEncoder
from weka.classifiers import Classifier
from weka.core.dataset import Attribute
from weka.core.converters import ndarray_to_instances
import weka.core.jvm as jvm


class SklearnWekaWrapper(object):

    def __init__(self, classifier_name):

        # Defaults
        class_name = 'weka.classifiers.trees.RandomForest'
        options = None

        if classifier_name == 'wrf':
            class_name = 'weka.classifiers.trees.RandomForest'
            options = None
        elif classifier_name == 'wj48':
            class_name = 'weka.classifiers.trees.J48'
            options = None
        elif classifier_name == 'wnb':
            class_name = 'weka.classifiers.bayes.NaiveBayes'
            options = '-D'
        elif classifier_name == 'wbn':
            class_name = 'weka.classifiers.bayes.BayesNet'
            options = '-D -Q weka.classifiers.bayes.net.search.local.TAN -- -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5'
        elif classifier_name == 'wsv':
            # Implementation of one-class SVM used in Anomaly Detection mode
            class_name = 'weka.classifiers.functions.LibSVM'
            options = '-S 2'

        if options is not None:
            self._classifier = Classifier(classname=class_name, options=[option for option in options.split()])
        else:
            self._classifier = Classifier(classname=class_name)

    def fit(self, training_set, ground_truth):

        self.ground_truth = ground_truth

        training_set = self._sklearn2weka(training_set, self.ground_truth)
        training_set.class_is_last()

        t = time.time()
        self._classifier.build_classifier(training_set)
        t = time.time() - t

        self.t_ = t

        return self

    def predict(self, testing_set):

        testing_set = self._sklearn2weka(testing_set, self.oracle)
        testing_set.class_is_last()

        preds = []
        for index, inst in enumerate(testing_set):
            pred = self._classifier.classify_instance(inst)
            preds.append(pred)

        preds = np.vectorize(self._dict.get)(preds)

        return np.array(preds)

    def predict_proba(self, testing_set):

        testing_set = self._sklearn2weka(testing_set, self.oracle)
        testing_set.class_is_last()

        dists = []
        for index, inst in enumerate(testing_set):
            dist = self._classifier.distribution_for_instance(inst)
            dists.append(dist)

        return np.array(dists)

    def set_oracle(self, oracle):

        self.oracle = oracle

    def _sklearn2weka(self, features, labels=None):

        # All weka datasets have to be a zero-based coding for the column of labels
        # We can use non-aligned labels for training and testing because the labels
        # in testing phase are only used to obtain performance, but not for preds.
        # We compute performance off-line.
        labels_encoder = OrdinalEncoder()
        labels_nominal = labels_encoder.fit_transform(np.array(labels).reshape(-1, 1))

        labels_column = np.reshape(labels_nominal, [labels_nominal.shape[0], 1])

        # TODO: find another way to do the same
        # The follow is used to assign the value of _dict only in training phase
        if not hasattr(self, '_dict') and labels is not None:

            dict = {}

            for label, nominal in zip(labels, labels_nominal):
                if nominal.item(0) not in dict:
                    dict[nominal.item(0)] = label

            self._dict = dict

        weka_dataset = ndarray_to_instances(np.ascontiguousarray(features, dtype=np.float_), 'weka_dataset')
        weka_dataset.insert_attribute(Attribute.create_nominal('tag', [str(float(i)) for i in range(len(self._dict))]),
                                      features.shape[1])

        if labels is not None:
            for index, inst in enumerate(weka_dataset):
                inst.set_value(features.shape[1], labels_column[index])
                weka_dataset.set_instance(index, inst)

        return weka_dataset
