import numpy as np
import time
import random

from sklearn.model_selection import StratifiedKFold

from ..models.anomaly_detector import AnomalyDetector


class SuperLearnerClassifier(object):
    def __init__(self, first_level_learners, num_classes, anomaly_class=None, features_number=0):

        self.num_classes = num_classes
        self.first_level_classifiers = []

        for first_level_learner in first_level_learners:
            if first_level_learner.startswith('s') or first_level_learner.startswith('k'):
                self.first_level_classifiers.append(
                    AnomalyDetector(first_level_learner, anomaly_class, features_number))

        self.anomaly_class = anomaly_class

        if self.anomaly_class == 1:
            self.normal_class = 0
        else:
            self.normal_class = 1

        self.first_level_classifiers_number = len(self.first_level_classifiers)

        self.proba = None

    def fit(self, dataset, labels):

        t = time.time()
        # First: split train set of node of hyerarchy in 20% train and 80% testing
        # with respective ground truth NB: for first training of first level models
        # SuperLearner use only training set
        # training_set, _, ground_truth, _ = train_test_split(dataset, labels, train_size=0.2)

        # Second: fit classifiers of first level through train set
        # proba, true = self._first_level_fit(training_set, ground_truth)

        # Three: fit combiner passing it obtained probabilities and respective ground truth
        # this phase return weights, i.e. decisions template, per each classifier
        # self.weights = self._meta_learner_fit(proba, true)

        # Four: final fit of first level classifier consist in fitting all models
        # with all the dataset, that is the train set of the node of hierarchy
        self._first_level_final_fit(dataset, labels)

        t = time.time() - t

        self.t_ = t

        return self

    def _first_level_fit(self, dataset, labels):

        # First level fit consist in a stratified ten-fold validation
        # to retrieve probabilities associated at the ground truth
        # NB: in input there is the train set, in output probabilities associated to this set
        # NB2: we don't need the predictions
        skf = StratifiedKFold(n_splits=10, shuffle=True)

        oracles_per_fold = []
        # predictions_per_fold = []
        probabilities_per_fold = []

        for train_index, test_index in skf.split(dataset, labels):

            training_set = dataset[train_index]
            testing_set = dataset[test_index]
            ground_truth = labels[train_index]
            oracle = labels[test_index]

            # prediction = np.ndarray(shape=[len(test_index), self.first_level_classifiers_number],dtype=int)
            probability = np.ndarray(shape=[len(test_index), self.first_level_classifiers_number], dtype=object)

            for i, first_level_classifier in enumerate(self.first_level_classifiers):
                first_level_classifier.fit(training_set, ground_truth)

                first_level_classifier.set_oracle(oracle)
                # prediction[:, i] = first_level_classifier.predict(testing_set).tolist()
                probability[:, i] = first_level_classifier.predict_proba(testing_set).tolist()

            oracles_per_fold.append(oracle)
            # predictions_per_fold.append(prediction)
            probabilities_per_fold.append(probability)

        probabilities = np.concatenate(probabilities_per_fold)
        oracles = np.concatenate(oracles_per_fold)

        return probabilities, oracles

    def _meta_learner_fit(self, dataset, labels):

        # TODO: fit the aggregation algorithm

        # if len(dataset.shape) == 1: # Hard combiner

        # elif len(dataset.shape) == 2: # Soft combiner

        # We use Decision_Template as combiner
        decision_template = self._fit_Decision_Template(np.transpose(dataset), labels)

        return decision_template

    # else: # Error
    #     raise Exception

    # pass

    def _fit_Decision_Template(self, decision_profile_list, validation_labels):

        vect_cont = [0] * self.num_classes
        decision_template = np.zeros(
            (self.num_classes, len(decision_profile_list), self.num_classes))  # Decision Template [LxKxL]

        dp = []  # lista dei profili di decisione corrispondenti ad ogni biflusso
        dp_temp = []  # lista temporanea usata per costruire un decision profile a partire dai sotf output dei classificatori

        # costruzione lista decision profile
        for i in range(len(decision_profile_list[0])):  # ciclo sulle predizioni dei classificatori
            for j in range(len(decision_profile_list)):  # ciclo sui classificatori
                dp_temp.append(decision_profile_list[j][i])
            dp.append(dp_temp)
            dp_temp = []

        # Calcolo Decision Template(Decision Profile medi)
        for i in range(len(validation_labels)):  # ciclo sulla ground truth del validation set
            # try:

            decision_template[validation_labels[i]] = np.add(decision_template[validation_labels[i]], dp[i])
            vect_cont[validation_labels[i]] += 1
        # except:
        #     print(set(validation_labels))
        #     print(decision_template.shape)
        #     exit(1)

        for i in range(decision_template.shape[0]):  # Ciclo sul numero di classi
            decision_template[i] = decision_template[i] / vect_cont[i]

        return decision_template

    def _first_level_final_fit(self, dataset, labels):

        for first_level_classifier in self.first_level_classifiers:
            first_level_classifier.fit(dataset, labels)

    def predict(self, testing_set):

        # First: we need prediction of first level models
        # first_proba = self._first_level_predict_proba(testing_set)
        first_pred = self._first_level_predict(testing_set)
        # Second: we pass first level prediction to combiner
        pred = self._meta_learner_predict(first_pred)
        # Third: for consistency, predict returns only predictions
        # NB: probabilities are not wasted, but saved in order to return
        # in case of calling predict_proba function
        # NB2: if you measure testing time, you have to comment next row
        # self.proba = proba

        return pred

    def _first_level_predict(self, testing_set):

        predictions = []

        for first_level_classifier in self.first_level_classifiers:
            first_level_classifier.set_oracle(self.oracle)
            pred = first_level_classifier.predict(testing_set)

            predictions.append(pred)

        return predictions

    def _first_level_predict_proba(self, testing_set):

        probabilities = []

        for first_level_classifier in self.first_level_classifiers:
            first_level_classifier.set_oracle(self.oracle)
            proba = first_level_classifier.predict_proba(testing_set)

            probabilities.append(proba)

        return probabilities

    def _meta_learner_predict(self, testing_set):

        # predictions = self._predict_Decision_Template(testing_set, self.weights)

        predictions = self._MV(testing_set)

        return predictions

    def _predict_Decision_Template(self, decision_profile_list, decision_template):

        mu_DT_SE = [0] * self.num_classes  # vettore delle "distanze" tra  il Decision Profile e il Decision Template
        decision = []
        soft_comb_values = []

        dp_predict = []
        dp_temp_predict = []

        for i in range(len(decision_profile_list[0])):  # ciclo sulle predizioni dei classificatori
            for j in range(len(decision_profile_list)):  # ciclo sui classificatori
                dp_temp_predict.append(decision_profile_list[j][i])
            dp_predict.append(dp_temp_predict)
            dp_temp_predict = []

        dp_predict = np.asarray(dp_predict, dtype=int)

        # Distanza Euclidea quadratica
        for i in range(len(dp_predict)):
            for j in range(self.num_classes):  # ciclo sulle classi
                mu_DT_SE[j] = 1 - (1 / (float(self.num_classes) * (len(dp_predict[0])))) * (
                        np.linalg.norm(decision_template[j] - dp_predict[i]) ** 2)

            soft_comb_values.append(mu_DT_SE)

            # tie-Braking
            if mu_DT_SE.count(max(mu_DT_SE)) < 2:
                decision_class = mu_DT_SE.index(max(mu_DT_SE))
            else:
                decision_class = random.choice([i for i, x in enumerate(mu_DT_SE) if x == max(mu_DT_SE)])

            mu_DT_SE = [0] * self.num_classes
            decision.append(decision_class)

        # soft value per le filtered performance
        soft_comb_values = np.asarray(soft_comb_values)

        x_min = soft_comb_values.min()
        x_max = soft_comb_values.max()

        for i in range(soft_comb_values.shape[0]):
            for j in range(soft_comb_values.shape[1]):
                soft_comb_values[i][j] = (soft_comb_values[i][j] - x_min) / (x_max - x_min)

        return decision, soft_comb_values

    def _mode(self, array):
        '''
        Returns the (multi-)mode of an array
        '''

        most = max(list(map(array.count, array)))
        return list(set(filter(lambda x: array.count(x) == most, array)))

    def _MV(self, predictions_list):
        '''
        Implements majority voting combination rule.
        :param predictions_list: i-th list contains the predictions of i-th classifier
        '''

        import random

        sample_predictions_list = list(zip(*predictions_list))

        MV_predictions = []
        for sample_predictions in sample_predictions_list:
            modes = self._mode(sample_predictions)
            MV_predictions.append(random.choice(modes))

        return np.asarray(MV_predictions, dtype=int)

    def predict_proba(self, testing_set):

        if self.proba is None:
            self.predict(testing_set)

        return self.proba

    def set_oracle(self, oracle):

        self.oracle = oracle

        print(np.unique(self.oracle, return_counts=True))
