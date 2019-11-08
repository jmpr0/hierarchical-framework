supported_classifiers = {
    'srf': 'Sklearn_RandomForest',
    'scr': 'Sklearn_CART',  # C4.5
    'drf': 'Spark_RandomForest',
    'dnb': 'Spark_NaiveBayes',
    'dmp': 'Spark_MultilayerPerceptron',
    'dgb': 'Spark_GBT',
    'ddt': 'Spark_DecisionTree',
    'wnb': 'Weka_NaiveBayes',
    'wbn': 'Weka_BayesNetwork',
    'wrf': 'Weka_RandomForest',
    'wj48': 'Weka_J48',  # C4.5
    'wsl': 'Weka_SuperLearner',
    'kdae': 'Keras_StackedDeepAutoencoderClassifier',
    'kmlp': 'Keras_MultiLayerPerceptronClassifier',
    'dkdae': 'Spark_Keras_StackedDeepAutoencoderClassifier',
    'dkmlp': 'Spark_Keras_MultiLayerPerceptronClassifier'
}

supported_detectors = {
    'ssv': 'Sklearn_OC-SVM',
    'sif': 'Sklearn_IsolationForest',  # C4.5
    'slo': 'Sklearn_LocalOutlierFactor',
    'ssv1': 'Sklearn_OC-SVM',
    'sif1': 'Sklearn_IsolationForest',  # C4.5
    'slo1': 'Sklearn_LocalOutlierFactor',
    'ssv2': 'Sklearn_OC-SVM',
    'slo2': 'Sklearn_LocalOutlierFactor',
    'ksae': 'Keras_StackedAutoencoder',
    'kdae': 'Keras_DeepAutoencoder',
    'kc2dae': 'Keras_Convolutional2DAutoencoder'
}
