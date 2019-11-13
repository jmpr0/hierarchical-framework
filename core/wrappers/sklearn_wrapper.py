import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.covariance import EllipticEnvelope
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import QuantileTransformer
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import FactorAnalysis
from sklearn.multiclass import OutputCodeClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.model_selection import GridSearchCV


class MinMaxScalerExpPenalty(MinMaxScaler):
    def __init__(self, feature_range=(0, 1), copy=True):
        super(MinMaxScalerExpPenalty, self).__init__(
            feature_range=feature_range, copy=copy
        )

    def transform_exppenalty(self, X):
        X = self.transform(X)
        X[np.where(X < 0)] = X[np.where(X < 0)] - np.exp(-X[np.where(X < 0)]) + 1
        X[np.where(X > 1)] = X[np.where(X > 1)] + np.exp(X[np.where(X > 1)] - 1) - 1
        return X


class OneClassSVMAnomalyDetector(OneClassSVM):
    def __init__(self, kernel='rbf', degree=3, gamma='auto_deprecated', coef0=0.0, tol=0.001, nu=0.5,
                 shrinking=True, cache_size=200, verbose=False, max_iter=-1, random_state=None):
        super(OneClassSVMAnomalyDetector, self).__init__(
            kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol, nu=nu,
            shrinking=shrinking, cache_size=cache_size, verbose=verbose,
            max_iter=max_iter, random_state=random_state
        )

    def score(self, X, y=None):
        '''
        OneClassSVM decision function returns signed distance to the separating hyperplane.
        This distance results positive for inlier and negative for outlier.
        OneClassSVM learns the hyperplane that define soft boundaries of training samples,
        and returns the outlierness score for each sample.
        In this work we optimize OneClassSVM hyperparameters with an heuripstic that borns
        from the passing of solely benign examples, i.e. we want the hyperparameters combination
        that maximize the score (higher the less abnormal) and minimize the negative scores.
        This can be obtained by maximizing the mean of positive scores,
        and minimizing (maximizing) the ratio of negative (positives) ones.
        '''
        # decisions = self.decision_function(X)
        # true_decisions = decisions[ np.where(decisions >= 0) ]
        # if len(true_decisions) > 0:
        #     true_decisions = MinMaxScaler().fit_transform(true_decisions.reshape(-1,1))
        #     factor = len(true_decisions)/len(decisions)
        #     return np.mean(true_decisions)*factor
        # else:
        #     return 0
        decisions = self.decision_function(X)
        # print(np.isinf(decisions).any())
        if np.isinf(decisions).all():
            return 0
        decisions = MinMaxScaler().fit_transform(decisions.reshape(-1, 1))
        return np.mean(decisions)


class IsolationForestAnomalyDetector(IsolationForest):
    def __init__(self, n_estimators=100, max_samples='auto', contamination='legacy', max_features=1.0,
                 bootstrap=False, n_jobs=None, behaviour='old', random_state=None, verbose=0):
        super(IsolationForestAnomalyDetector, self).__init__(
            n_estimators=n_estimators, max_samples=max_samples, contamination=contamination,
            max_features=max_features, bootstrap=bootstrap, n_jobs=n_jobs, behaviour=behaviour,
            random_state=random_state, verbose=verbose
        )

    def score(self, X, y=None):
        '''
        IsolationForest decision function returns the anomaly score for each sample, negative for outliers and positive for inliers. This score is the measure of
        normality of an observation and is equivalent to the number of the split of the tree to isolate the particular
        sample. During optimization phase we set the contamination factor (portion of outliers in the dataset) to zero,
        because we are training only over benign samples. To found optimal hyperparameters configuration we had to
        maximize the mean of positive scores and minimize the negatives ones. So the approach is the same we use to optimize OC-SVM.
        '''
        # decisions = self.decision_function(X)
        # true_decisions = decisions[ np.where(decisions >= 0) ]
        # if len(true_decisions) > 0:
        #     true_decisions = MinMaxScaler().fit_transform(true_decisions.reshape(-1,1))
        #     factor = len(true_decisions)/len(decisions)
        #     return np.mean(true_decisions)*factor
        # else:
        #     return 0
        decisions = self.decision_function(X)
        if np.isinf(decisions).all():
            # decisions[np.where(np.isinf(decisions))] = np.nanmax(decisions[np.where(np.isfinite(decisions))])
            return 0
        decisions = MinMaxScaler().fit_transform(decisions.reshape(-1, 1))
        return np.mean(decisions)


class LocalOutlierFactorAnomalyDetector(LocalOutlierFactor):
    def __init__(self, n_neighbors=20, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None,
                 contamination='legacy', novelty=False, n_jobs=None):
        super(LocalOutlierFactorAnomalyDetector, self).__init__(
            n_neighbors=n_neighbors, algorithm=algorithm, leaf_size=leaf_size, metric=metric,
            p=p, metric_params=metric_params, contamination=contamination, novelty=novelty, n_jobs=n_jobs
        )
        self.n_neighbors = n_neighbors

    def score(self, X, y=None):
        '''
        LOF decision function return the shifted opposite of the local outlier factor of samples. The greater the more normal.
        To train this algorithm we cannot set contamination to 0, because zero is the -inf asymptote. So we set contamination
        to the minimum permitted value (e.g. 1e-4).
        The optimization of hyperparameters for LOF is slightly diverse w.r.t. the used in previous models.
        In particular, we only maximize mean of scores, after min max normalization.
        '''
        # decisions = self.decision_function(X)
        # true_decisions = decisions[ np.where(decisions >= 0) ]
        # if len(true_decisions) > 0:
        #     true_decisions = MinMaxScaler().fit_transform(true_decisions.reshape(-1,1))
        #     factor = len(true_decisions)/len(decisions)
        #     return 1/np.std(true_decisions)*factor
        # else:
        #     return 0
        decisions = self.decision_function(X)
        if np.isinf(decisions).all():
            return 0
        decisions = MinMaxScaler().fit_transform(decisions.reshape(-1, 1))
        return np.mean(decisions)


class MixtureLocalizationOutliers(object):
    def __init__(self, n_components=2):
        self.GMM = GaussianMixture(n_components)
        self.LOF = LocalOutlierFactor(n_neighbors=2, novelty=True, contamination=1e-4)

        self.decisions = None

    def fit(self, X, y=None):
        pdfs = self.GMM.fit_predict(X)
        self.LOF.fit(pdfs)
        lofs = self.LOF.decision_function(pdfs)
        self.lower_lof, self.upper_lof = np.percentile(lofs, [.25, .75])

    def predict(self, X):
        pdfs = self.GMM.predict(X)
        lofs = self.LOF.decision_function(pdfs)
        preds = []
        for pdf, lof in zip(pdfs, lofs):
            if lof <= self.lower_lof or pdf >= self.upper_lof:
                preds.append(-1)
            else:
                preds.append(1)
        self.decisions = lofs
        return preds

    def decision_function(self, X):
        if self.decisions is None:
            self.predict(X)
        return self.decisions
