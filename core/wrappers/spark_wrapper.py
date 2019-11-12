import pandas
import numpy as np
import time
from scipy.sparse import issparse

from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import udf
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.linalg import Vectors
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler

# import_distkeras
from distkeras.trainers import ADAG
from distkeras.predictors import ModelPredictor
from distkeras.transformers import OneHotTransformer

from .keras_wrapper import SklearnKerasWrapper
import core.utils.preprocessing


class SingletonSparkSession(object):
    __spark = None

    @staticmethod
    def get_session(raw_conf=None):
        if SingletonSparkSession.__spark is None:
            SingletonSparkSession(raw_conf)
        return SingletonSparkSession.__spark

    def __init__(self, raw_conf):
        if SingletonSparkSession.__spark is not None:
            raise Exception('The SingletonSparkSession class is a singleton.')
        else:
            conf = SparkConf().setAll(raw_conf)
            sc = SparkContext(conf=conf)
            sc.setLogLevel("ERROR")
            __spark = SparkSession(sc)


class SklearnSparkWrapper(object):

    def __init__(self, classifier_class, num_classes=None, numerical_features_index=None, nominal_features_index=None,
                 fine_nominal_features_index=None, classifier_opts=None, epochs_number=None, level=None, fold=None,
                 classify=None, workers_number=None, arbitrary_discr='', weight_features=True):

        self.spark_session = SingletonSparkSession.get_session()
        self.scale = False
        self.probas_ = None
        self.is_keras = False
        self.workers_number = workers_number
        self.epochs_number = epochs_number

        if classifier_class == 'drf':
            self._classifier = RandomForestClassifier(featuresCol='features', labelCol='categorical_label',
                                                      predictionCol='prediction', probabilityCol='probability',
                                                      rawPredictionCol='rawPrediction',
                                                      maxDepth=20, maxBins=128, minInstancesPerNode=1,
                                                      minInfoGain=0.0, maxMemoryInMB=1024, cacheNodeIds=False,
                                                      checkpointInterval=10,
                                                      impurity='gini', numTrees=100, featureSubsetStrategy='sqrt',
                                                      seed=None, subsamplingRate=1.0)
        elif classifier_class == 'dnb':
            self._classifier = NaiveBayes(featuresCol='scaled_features', labelCol='categorical_label',
                                          predictionCol='prediction', probabilityCol='probability',
                                          rawPredictionCol='rawPrediction', smoothing=1.0,
                                          modelType='multinomial', thresholds=None, weightCol=None)
            self.scale = True
        # elif classifier_class == 'dmp':
        #     layers = []
        #     input_dim = numerical_features_length + np.sum(nominal_features_lengths)
        #     depth = classifier_opts[0]
        #     if input_dim is not None and num_classes is not None:
        #         layers = [input_dim] + [100 for _ in range(int(depth))] + [num_classes]
        #     else:
        #         raise Exception('Both input_dim and num_classes must be declared.')
        #     self._classifier = MultilayerPerceptronClassifier(featuresCol='scaled_features',
        #                                                       labelCol='categorical_label',
        #                                                       predictionCol='prediction', maxIter=100, tol=1e-06,
        #                                                       seed=0, layers=layers, blockSize=32, stepSize=0.03,
        #                                                       solver='l-bfgs',
        #                                                       initialWeights=None, probabilityCol='probability',
        #                                                       rawPredictionCol='rawPrediction')
        #     self.scale = True
        elif classifier_class == 'dgb':
            self._classifier = GBTClassifier(featuresCol='features', labelCol='categorical_label',
                                             predictionCol='prediction', maxDepth=5, maxBins=32,
                                             minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=256,
                                             cacheNodeIds=False, checkpointInterval=10, lossType='logistic',
                                             maxIter=20, stepSize=0.1, seed=None,
                                             subsamplingRate=1.0, featureSubsetStrategy='all')
        elif classifier_class == 'ddt':
            self._classifier = DecisionTreeClassifier(featuresCol='features', labelCol='categorical_label',
                                                      predictionCol='prediction', probabilityCol='probability',
                                                      rawPredictionCol='rawPrediction', maxDepth=5, maxBins=32,
                                                      minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=256,
                                                      cacheNodeIds=False, checkpointInterval=10, impurity='gini',
                                                      seed=None)
        elif classifier_class.startswith('dk'):
            depth = classifier_opts[0]
            self.keras_wrapper = SklearnKerasWrapper(*classifier_opts, model_class=classifier_class[1:],
                                                     epochs_number=epochs_number, num_classes=num_classes,
                                                     nominal_features_index=[], fine_nominal_features_index=[],
                                                     numerical_features_index=numerical_features_index + fine_nominal_features_index + nominal_features_index,
                                                     level=level, fold=fold, classify=classify,
                                                     weight_features=weight_features, arbitrary_discr=arbitrary_discr)
            self._classifier = self.keras_wrapper.init_model()[2]
            self.nominal_features_index = nominal_features_index
            self.is_keras = True

    def fit(self, training_set, ground_truth):

        core.utils.preprocessing.ohe(training_set, self.nominal_features_index)

        self.ground_truth = ground_truth

        if self.is_keras:
            nom_training_set, num_training_set = self.keras_wrapper.split_nom_num_features(training_set)
            training_set = np.array([num_training_set] + nom_training_set)
            training_set = self._sklearn2spark(training_set, self.ground_truth, True)
            self._classifier = ADAG(keras_model=self._classifier, worker_optimizer='adadelta',
                                    loss='categorical_crossentropy', num_workers=self.workers_number,
                                    batch_size=32, communication_window=12, num_epoch=self.epochs_number,
                                    features_col='features', label_col='categorical_label',
                                    metrics=['categorical_accuracy'])
        else:
            training_set = core.utils.preprocessing.sparse_flattening(training_set)
            training_set = self._sklearn2spark(training_set, self.ground_truth)
        # print(self.ground_truth)
        # input()
        if self.scale:
            scaler = MinMaxScaler(inputCol='features', outputCol='scaled_features')
            self.scaler_model_ = scaler.fit(training_set)
            scaled_training_set = self.scaler_model_.transform(training_set)
        else:
            scaled_training_set = training_set

        # Maybe keras model train is returned
        if not self.is_keras:
            t = time.time()
            self.model = self._classifier.fit(scaled_training_set)
            t = time.time() - t
        else:
            t = time.time()
            self.model = self._classifier.train(scaled_training_set)
            t = time.time() - t
            self.model = ModelPredictor(keras_model=self.model, features_col='features')

        self.t_ = t

        return self

    def predict(self, testing_set):

        core.utils.preprocessing.ohe(testing_set, self.nominal_features_index)

        if self.is_keras:
            nom_testing_set, num_testing_set = self.keras_wrapper.split_nom_num_features(testing_set)
            testing_set = [num_testing_set] + nom_testing_set
        else:
            testing_set = core.utils.preprocessing.sparse_flattening(testing_set)

        testing_set = self._sklearn2spark(testing_set)
        if self.scale:
            scaled_testing_set = self.scaler_model_.transform(testing_set)
        else:
            scaled_testing_set = testing_set

        if not self.is_keras:
            self.results = self.model.transform(scaled_testing_set)
            preds = np.array([int(float(row.prediction)) for row in self.results.collect()])
            self.probas_ = np.array([row.probability.toArray() for row in self.results.collect()])
        else:
            preds = self.model.predict(testing_set)

        return preds

    def predict_proba(self, testing_set):

        if self.probas_ is None:
            self.predict(testing_set)
        return np.array(self.probas_)

    def set_oracle(self, oracle):

        self.oracle = oracle

    def _sklearn2spark(self, features, labels=None, multi_input=False):

        features_names = []
        if multi_input:
            dataset = pandas.DataFrame()
            c = 0
            for i, feature in enumerate(features):
                feature = feature.toarray().T.tolist() if issparse(feature) else feature.T.tolist()
                for f in feature:
                    dataset['features_%s' % c] = f
                    features_names.append('features_%s' % c)
                    c += 1
            dataset['categorical_label'] = labels if labels is not None else [''] * features.shape[0]
        else:
            dataset = pandas.DataFrame(
                {'features': features.tolist(),
                 'categorical_label': labels if labels is not None else [''] * features.shape[0]
                 })

        spark_dataset_with_list = self.spark_session.createDataFrame(dataset)
        if multi_input:
            # Join all features columns
            assembler = VectorAssembler(inputCols=features_names, outputCol='features')
            assembler.setHandleInvalid('skip')
            spark_dataset_with_list = assembler.transform(spark_dataset_with_list)
            # Join all labels columns
            onehotencoder = OneHotTransformer(output_dim=len(np.unique(labels)), input_col='categorical_label',
                                              output_col='ohe_label')
            spark_dataset_with_list = onehotencoder.transform(spark_dataset_with_list)

        list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())

        if multi_input:
            spark_dataset = spark_dataset_with_list.select(
                list_to_vector_udf(spark_dataset_with_list['features']).alias('features'),
                spark_dataset_with_list['ohe_label'].alias('categorical_label')
            )
        else:
            spark_dataset = spark_dataset_with_list.select(
                list_to_vector_udf(spark_dataset_with_list['features']).alias('features'),
                spark_dataset_with_list['categorical_label']
            )

        # if spark_dataset.rdd.getNumPartitions() != self.workers_number:
        #     spark_dataset = spark_dataset.coalesce(numPartitions=self.workers_number)

        return spark_dataset
