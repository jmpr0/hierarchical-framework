import os
import numpy as np
import time
import gc
from copy import copy
from scipy.sparse import issparse
from scipy.sparse import csr_matrix

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
import core.utils.preprocessing

# For reproducibility
from tensorflow import set_random_seed, logging

set_random_seed(0)
logging.set_verbosity(logging.ERROR)

import keras
import cython
from keras import optimizers
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv1D
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import BatchNormalization
from keras.layers import LSTM
from keras.layers import AveragePooling1D
from keras.layers import MaxPooling1D
from keras.layers import UpSampling1D
from keras.layers import Activation
from keras.activations import relu
from keras.activations import elu
from keras.activations import sigmoid
from keras.activations import hard_sigmoid
from keras.activations import tanh
from keras.activations import softmax
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.losses import mean_squared_error
from keras.losses import categorical_crossentropy
from keras import backend as K
from keras.optimizers import SGD
from keras.optimizers import Adadelta
from keras.layers import concatenate
from keras.utils.vis_utils import plot_model
from keras import regularizers
from keras.layers import Lambda
from keras.models import clone_model


class SklearnKerasWrapper(BaseEstimator, ClassifierMixin):

    def __init__(self, depth='3', hidden_activation_function_name='elu', mode='n', sharing_ray='3', grafting_depth='0',
                 compression_ratio='.1', model_class='kdae', epochs_number=10, num_classes=1,
                 nominal_features_index=None, fine_nominal_features_index=None, numerical_features_index=None, fold=0,
                 level=0, classify=False, weight_features=False,
                 arbitrary_discr=''):

        # print(depth, hidden_activation_function, mode, sharing_ray, grafting_depth, compression_ratio,
        # model_class, epochs_number, num_classes, fold, level, classify)
        # input()
        model_discr = model_class + '_' + '_'.join(
            [str(c) for c in
             [depth, hidden_activation_function_name, mode, sharing_ray, grafting_depth, compression_ratio]]
        )

        self.sharing_ray = int(sharing_ray)
        self.grafting_depth = int(grafting_depth)
        self.depth = int(depth)
        self.compression_ratio = float(compression_ratio)
        self.sparse = False
        self.variational = False
        self.denoising = False
        self.propagational = False
        self.auto = False
        self.mode = mode
        self.fold = fold
        self.level = level
        self.epochs_number = epochs_number
        self.patience = 1
        self.min_delta = 1e-6
        if 'n' not in mode:
            if 'v' in mode:
                self.variational = True
                self.patience = self.epochs_number
            if 'd' in mode:
                self.denoising = True
                self.patience = self.epochs_number
            if 's' in mode:
                self.sparse = True
                self.patience = 10
            if 'p' in mode:
                self.propagational = True
            if 'a' in mode:
                self.auto = True
                model_discr = model_class + '_' + '_'.join(
                    [str(c) for c in [depth, hidden_activation_function_name, mode]]
                )
        self.hidden_activation_function_name = hidden_activation_function_name
        self.hidden_activation_function = globals()[hidden_activation_function_name]
        self.model_class = model_class
        self.numerical_features_length = None
        self.nominal_features_lengths = None
        self.num_classes = num_classes
        self.nominal_features_index = nominal_features_index
        self.fine_nominal_features_index = fine_nominal_features_index
        self.numerical_features_index = numerical_features_index
        self.classify = classify
        self.weight_features = weight_features
        log_folder = './data_%s/material/log' % model_discr
        plot_folder = './data_%s/material/plot' % model_discr
        summary_folder = './data_%s/material/summary' % model_discr
        tmp_folder = './data_%s/tmp' % model_discr
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        if not os.path.exists(summary_folder):
            os.makedirs(summary_folder)
        if not os.path.exists(tmp_folder):
            os.makedirs(tmp_folder)
        self.log_file = '%s/%s_%s_model_%s_l_%s_log.csv' % (log_folder, arbitrary_discr, fold, model_discr, level + 1)
        self.plot_file = '%s/%s_%s_model_%s_l_%s_plot' % (plot_folder, arbitrary_discr, fold, model_discr, level + 1)
        self.summary_file = '%s/%s_%s_model_%s_l_%s_summary' % (
            summary_folder, arbitrary_discr, fold, model_discr, level + 1)
        self.losses_file = '%s/%s_%s_model_%s_l_%s_losses.dat' % (
            summary_folder, arbitrary_discr, fold, model_discr, level + 1)
        self.checkpoint_file = "%s/%s_weights.hdf5" % (tmp_folder, arbitrary_discr)
        self.scaler = QuantileTransformer()
        self.label_encoder = OneHotEncoder()
        self.proba = None
        # For internal optimization
        self.k_internal_fold = 3
        self.complexity_red = 1

        self.numerical_output_activation_function = 'sigmoid'
        self.numerical_loss_function = 'mean_squared_error'
        self.nominal_output_activation_functions = None
        self.nominal_loss_functions = None

        if self.fine_nominal_features_index is not None:
            self.nominal_output_activation_functions = []
            self.nominal_loss_functions = []
            for index in sorted(self.nominal_features_index + self.fine_nominal_features_index):
                if index in self.nominal_features_index:
                    self.nominal_output_activation_functions.append('softmax')
                    self.nominal_loss_functions.append('categorical_crossentropy')
                elif index in self.fine_nominal_features_index:
                    self.nominal_output_activation_functions.append('sigmoid')
                    self.nominal_loss_functions.append('mean_squared_error')

    def fit(self, X, y=None):

        if not -1 <= self.sharing_ray <= self.depth or not 0 <= self.grafting_depth <= self.depth + 1:
            raise Exception('Invalid value for Shared Ray or for Grafting Depth.\n\
                Permitted values are: S_r in [-1, Depth] and G_d in [0, Depth+1].')

        self.nominal_encoder = core.utils.preprocessing.ohe(X, self.nominal_features_index)
        self.numerical_features_length, self.nominal_features_lengths = core.utils.preprocessing.get_num_nom_lengths(X)

        nom_X, num_X = self.split_nom_num_features(X)

        # If current parameters are not previously initialized, classifier would infer them from training set
        if self.nominal_features_lengths is None:
            self.nominal_features_lengths = [v[0][0].shape[1] for v in nom_X]
        if self.numerical_features_length is None:
            self.numerical_features_length = num_X.shape[1]
        if self.nominal_output_activation_functions is None:
            self.nominal_output_activation_functions = [
                'softmax' if np.array([np.sum(r) == 1 and np.max(r) == 1 for r in v]).all() else 'sigmoid' for v in
                nom_X]
        if self.nominal_loss_functions is None:
            self.nominal_loss_functions = ['categorical_crossentropy' if np.array(
                [np.sum(r) == 1 and np.max(r) == 1 for r in v]).all() else 'mean_squared_error' for v in nom_X]

        # Define weights for loss average
        if self.weight_features:
            self.numerical_weight = self.numerical_features_length
        else:
            self.numerical_weight = 1
        self.nominal_weights = [1] * len(self.nominal_features_lengths)

        reconstruction_models, _, classification_models, models_names = self.init_models(self.model_class,
                                                                                         self.num_classes,
                                                                                         self.classify)
        # Scaling training set for Autoencoder
        one_hot_y = None
        scaled_num_X = self.scaler.fit_transform(num_X)
        if self.classify:
            one_hot_y = self.label_encoder.fit_transform(y.reshape(-1, 1))
        normal_losses_per_fold = []
        times_per_fold = []
        if len(reconstruction_models) > 1:
            # Iterating over all the models to find the best
            for index in range(len(reconstruction_models)):
                print('\nStarting StratifiedKFold for %s\n' % models_names[index])
                skf = StratifiedKFold(n_splits=self.k_internal_fold, shuffle=True)
                normal_losses_per_fold.append([])
                times_per_fold.append([])
                f = 0
                reconstruction_model = reconstruction_models[index]
                classification_model = classification_models[index]
                for train_index, validation_index in skf.split(
                        np.zeros(shape=(
                                num_X.shape[0], self.numerical_features_length + len(self.nominal_features_lengths))),
                        y):
                    # To reduce amount of similar models to built, we save initial weights at first fold, than we will use this for subsequent folds.
                    if f == 0:
                        reconstruction_model.save_weights(self.checkpoint_file)
                    else:
                        reconstruction_model.load_weights(self.checkpoint_file)
                    f += 1
                    print('\nFold %s\n' % f)
                    # for _model in self._models:
                    if not self.denoising:
                        times_per_fold[-1].append(time.time())
                        # Train Autoencoder
                        train_history = \
                            reconstruction_model.fit(
                                [scaled_num_X[train_index, :]] + [nom_training[train_index, :] for nom_training in
                                                                  nom_X],
                                [scaled_num_X[train_index, :]] + [nom_training[train_index, :] for nom_training in
                                                                  nom_X],
                                epochs=self.epochs_number,
                                batch_size=32,
                                shuffle=True,
                                verbose=2,
                                # validation_split = 0.1,
                                validation_data=(
                                    [scaled_num_X[validation_index, :]] + [nom_training[validation_index, :] for
                                                                           nom_training in nom_X],
                                    [scaled_num_X[validation_index, :]] + [nom_training[validation_index, :] for
                                                                           nom_training in nom_X]
                                ),
                                callbacks=[
                                    EarlyStopping(monitor='val_loss', patience=self.patience, min_delta=self.min_delta),
                                    CSVLogger(self.log_file),
                                    # ModelCheckpoint(filepath=self.checkpoint_file, verbose=1, save_best_only=True)
                                ]
                            )
                        if self.classify:
                            for layer in reconstruction_model:
                                layer.trainable = False
                            train_history = \
                                classification_model.fit(
                                    [scaled_num_X[train_index, :]] + [nom_training[train_index, :] for nom_training in
                                                                      nom_X],
                                    one_hot_y[train_index, :],
                                    epochs=self.epochs_number,
                                    batch_size=32,
                                    shuffle=True,
                                    verbose=2,
                                    # validation_split = 0.1,
                                    validation_data=(
                                        [scaled_num_X[validation_index, :]] + [nom_training[validation_index, :] for
                                                                               nom_training in nom_X],
                                        one_hot_y[validation_index, :]
                                    ),
                                    callbacks=[
                                        EarlyStopping(monitor='val_loss', patience=self.patience,
                                                      min_delta=self.min_delta),
                                        CSVLogger(self.log_file),
                                        # ModelCheckpoint(filepath=self.checkpoint_file, verbose=1, save_best_only=True)
                                    ]
                                )
                    else:
                        noise_factor = .5
                        noisy_scaled_num_X = scaled_num_X[train_index, :] + noise_factor * np.random.normal(loc=.0,
                                                                                                            scale=1.,
                                                                                                            size=scaled_num_X[
                                                                                                                 train_index,
                                                                                                                 :].shape)
                        times_per_fold[-1].append(time.time())
                        # Train Autoencoder
                        train_history = \
                            reconstruction_model.fit(
                                [noisy_scaled_num_X] + [nom_training[train_index, :] for nom_training in nom_X],
                                [scaled_num_X[train_index, :]] + [nom_training[train_index, :] for nom_training in
                                                                  nom_X],
                                epochs=self.epochs_number,
                                batch_size=32,
                                shuffle=True,
                                verbose=2,
                                # validation_split = 0.1,
                                validation_data=(
                                    [scaled_num_X[validation_index, :]] + [nom_training[validation_index, :] for
                                                                           nom_training in nom_X],
                                    [scaled_num_X[validation_index, :]] + [nom_training[validation_index, :] for
                                                                           nom_training in nom_X]
                                ),
                                callbacks=[
                                    EarlyStopping(monitor='val_loss', patience=self.patience, min_delta=self.min_delta),
                                    CSVLogger(self.log_file),
                                    # ModelCheckpoint(filepath=self.checkpoint_file, verbose=1, save_best_only=True)
                                ]
                            )
                    # num_losses, nom_losses, clf_losses = self.get_losses(train_history.history,discr='val')
                    # # This order (i.e. num+nom) must be respected to compute weighthed average
                    # if self.classify or (len(num_losses) == 0 and len(nom_losses) == 0):
                    #     losses = clf_losses
                    #     normal_losses_per_fold[-1].append(losses)
                    # else:
                    #     losses = num_losses + nom_losses
                    #     # print(losses)
                    #     # print(num_weight + nom_weights)
                    #     normal_losses_per_fold[-1].append(np.mean(losses, weights=[ self.numerical_weight ]+self.nominal_weights))
                    normal_losses_per_fold[-1].append(self.get_loss(train_history.history, 'val_loss'))
                    times_per_fold[-1][-1] = time.time() - times_per_fold[-1][-1]
                # train_history = losses = nom_losses = num_losses = clf_losses = None
                # del train_history, losses, nom_losses, num_losses, clf_losses
                # gc.collect()
                normal_losses_per_fold[-1] = np.asarray(normal_losses_per_fold[-1])
                times_per_fold[-1] = np.asarray(times_per_fold[-1])
            mean_normal_losses = np.mean(normal_losses_per_fold, axis=1)
            std_normal_losses = np.std(normal_losses_per_fold, axis=1)
            # We use the sum of mean and std deviation to choose the best model
            meanPstd_normal_losses = mean_normal_losses + std_normal_losses
            mean_times = np.mean(times_per_fold, axis=1)
            # Saving losses per model
            with open(self.losses_file, 'a') as f:
                for mean_normal_loss, std_normal_loss, model_name in zip(mean_normal_losses, std_normal_losses,
                                                                         models_names):
                    f.write('%s %s %s\n' % (mean_normal_loss, std_normal_loss, model_name))
            min_loss = np.min(meanPstd_normal_losses)
            print(meanPstd_normal_losses)
            best_model_index = list(meanPstd_normal_losses).index(min_loss)
            best_model_name = models_names[best_model_index]
            t = np.max(mean_times)
            print(t)
        else:
            best_model_name = models_names[0]
        t = 0
        reconstruction_models = None
        classification_models = None
        del reconstruction_models
        del classification_models
        gc.collect()
        sharing_ray = int(best_model_name.split('_')[1])
        grafting_depth = int(best_model_name.split('_')[2])
        self.reconstruction_model_, _, self.classification_model_, _ = self.init_model(self.model_class, sharing_ray,
                                                                                       grafting_depth, self.num_classes,
                                                                                       self.classify)
        train_history = None
        if not self.denoising:
            t = time.time() - t
            if self.reconstruction_model_ is not None:
                # Train Autoencoder
                train_history = \
                    self.reconstruction_model_.fit(
                        [scaled_num_X] + nom_X,
                        [scaled_num_X] + nom_X,
                        epochs=self.epochs_number,
                        batch_size=32,
                        shuffle=True,
                        verbose=2,
                        callbacks=[
                            EarlyStopping(monitor='loss', patience=self.patience, min_delta=self.min_delta),
                            CSVLogger(self.log_file),
                            # ModelCheckpoint(filepath=self.checkpoint_file, verbose=1, save_best_only=True)
                        ]
                    )
                if self.classify:
                    for layer in self.reconstruction_model_.layers:
                        layer.trainable = False
            if self.classify:
                train_history = \
                    self.classification_model_.fit(
                        [scaled_num_X] + nom_X,
                        one_hot_y,
                        epochs=self.epochs_number,
                        batch_size=32,
                        shuffle=True,
                        verbose=2,
                        callbacks=[
                            EarlyStopping(monitor='loss', patience=self.patience, min_delta=self.min_delta),
                            CSVLogger(self.log_file),
                            # ModelCheckpoint(filepath=self.checkpoint_file, verbose=1, save_best_only=True)
                        ]
                    )
        # TODO: denoising training definition
        else:
            noise_factor = .5
            noisy_scaled_num_X = scaled_num_X[train_index, :] + noise_factor * np.random.normal(loc=.0, scale=1.,
                                                                                                size=scaled_num_X[
                                                                                                     train_index,
                                                                                                     :].shape)
            t = time.time() - t
            # Train Autoencoder
            train_history = \
                self.reconstruction_model_.fit(
                    [noisy_scaled_num_X] + nom_X,
                    [scaled_num_X] + nom_X,
                    epochs=self.epochs_number,
                    batch_size=32,
                    shuffle=True,
                    verbose=2,
                    callbacks=[
                        EarlyStopping(monitor='loss', patience=self.patience, min_delta=self.min_delta),
                        CSVLogger(self.log_file),
                        # ModelCheckpoint(filepath=self.checkpoint_file, verbose=1, save_best_only=True)
                    ]
                )
        # num_losses, nom_losses, clf_losses = self.get_losses(train_history.history)
        # # This order (i.e. num+nom) must be respected to compute weighthed average
        # if self.classify or (len(num_losses) == 0 and len(nom_losses) == 0):
        #     losses = clf_losses
        #     # self.normal_loss_ = np.mean(losses)
        # else:
        #     losses = num_losses + nom_losses
        # self.normal_loss_ = np.average(losses, weights = num_weight + nom_weights)
        # self.normal_loss_ = np.mean(losses)
        self.normal_loss_ = self.get_loss(train_history.history, 'loss')
        t = time.time() - t
        if self.reconstruction_model_ is not None:
            with open(self.summary_file + '_best_reconstruction_model_' + best_model_name + '.dat', 'w') as f:
                self.reconstruction_model_.summary(print_fn=lambda x: f.write(x + '\n'))
        if self.classify:
            with open(self.summary_file + '_best_classification_model_' + best_model_name + '.dat', 'w') as f:
                self.classification_model_.summary(print_fn=lambda x: f.write(x + '\n'))
        print('loss', self.normal_loss_)
        print('time', t)
        self.t_ = t
        return self

    def get_losses(self, history, discr=''):
        num_losses = []
        nom_losses = []
        clf_losses = []
        for loss_tag in history:
            if discr in loss_tag:
                if 'num' in loss_tag:
                    for num_loss in history[loss_tag][::-1]:
                        if not np.isnan(num_loss):
                            num_losses.append(num_loss)
                            break
                elif 'nom' in loss_tag:
                    for nom_loss in history[loss_tag][::-1]:
                        if not np.isnan(nom_loss):
                            nom_losses.append(nom_loss)
                            break
                elif loss_tag == 'loss':
                    for clf_loss in history[loss_tag][::-1]:
                        if not np.isnan(clf_loss):
                            clf_losses.append(clf_loss)
                            break

        return num_losses, nom_losses, clf_losses

    def get_loss(self, history, loss_tag):
        for loss in history[loss_tag][::-1]:
            if not np.isnan(loss):
                return loss
        return -1

    def get_weights(self, mod='r'):
        if 'r' in mod:
            return self.reconstruction_model_.get_weights()
        elif 'c' in mod:
            return self.classification_model_.get_weights()
        elif 'k' in mod:
            return self.compression_model.get_weights()
        return None

    def set_weights(self, weights, mod='r'):
        if 'r' in mod:
            self.reconstruction_model_.set_weights(weights)
            return self.reconstruction_model_
        elif 'c' in mod:
            self.classification_model_.set_weights(weights)
            return self.classification_model_
        elif 'k' in mod:
            self.compression_model.set_weights(weights)
            return self.compression_model
        return None

    def get_models(self, mod='rck'):
        models = []
        if 'r' in mod:
            models.append(self.reconstruction_model_)
        if 'c' in mod:
            models.append(self.classification_model_)
        if 'k' in mod:
            models.append(self.compression_model)
        if len(models) == 1:
            return models[0]
        return models

    def init_models(self, model_class=None, num_classes=None, classify=None):

        if model_class is None:
            model_class = self.model_class
        if num_classes is None:
            num_classes = self.num_classes
        if classify is None:
            classify = self.classify

        reconstruction_models = []
        compression_models = []
        classification_models = []
        models_n = []

        if self.auto:
            range_sr = range(0, self.depth + 1, self.complexity_red)
            range_gd = range(0, self.depth + 1, self.complexity_red)
        else:
            range_sr = [self.sharing_ray]
            range_gd = [self.grafting_depth]

        for sharing_ray in range_sr:
            for grafting_depth in range_gd:
                reconstruction_model, compression_model, classification_model, model_n = self.init_model(model_class,
                                                                                                         sharing_ray,
                                                                                                         grafting_depth,
                                                                                                         num_classes,
                                                                                                         classify)
                reconstruction_models.append(reconstruction_model)
                compression_models.append(compression_model)
                classification_models.append(classification_model)
                models_n.append(model_n)

        return reconstruction_models, compression_models, classification_models, models_n

    def init_model(self, model_class=None, sharing_ray=None, grafting_depth=None, num_classes=None, classify=None):

        if None in [self.nominal_features_lengths, self.numerical_features_length,
                    self.nominal_output_activation_functions, self.nominal_loss_functions]:
            raise Exception('Cannot call init_models or init_model methods of SklearnSparkWrapper without setting\
                nominal_features_lengths, numerical_features_length, nominal_output_activation_functions, and nominal_loss_functions\
                attributes. If you want to automatically initialize them, use SklearnSparkWrapper fit method.')

        if model_class is None:
            model_class = self.model_class
        if sharing_ray is None:
            sharing_ray = self.sharing_ray
        if grafting_depth is None:
            grafting_depth = self.grafting_depth
        if num_classes is None:
            num_classes = self.num_classes
        if classify is None:
            classify = self.classify

        if sharing_ray < 0 or sharing_ray > self.depth or grafting_depth < 0 or grafting_depth > self.depth:
            print('Inserted sharing ray or grafting depth are invalid.')
            exit(1)

        reduction_coeff = (1 - self.compression_ratio) / self.depth
        reduction_factors = [1.] + [1 - reduction_coeff * (i + 1) for i in range(self.depth)]
        # print(sharing_ray, grafting_depth)
        # print(reduction_coeff)
        # print(reduction_factors)
        latent_ratio = reduction_factors[-1]
        # This parameter is used to set the number of e/d layers nearest the middle layer to join
        # If it's more or equal than zero, the middle layer is automatically joined
        # The parameter take values in [0,self.depth], where 0 states no join
        # If it's more than zero, it indicates the couple of e/d layers to join, starting from central
        # sharing_ray = -1
        # This parameter specify at which level num model grafts on nom one
        # It takes values in [0,self.depth], where self.depth state for no graft, and others value refers to the 0,1,2,..self.depth-1 i-e/d layer
        # E.g. if it is equal to self.depth-1, the graft is done on last layer, if to 0, on the first encoder layer
        # grafting_depth = self.depth-1
        num_hidden_activation_function = self.hidden_activation_function
        nom_hidden_activation_function = self.hidden_activation_function
        hyb_activation_function = self.hidden_activation_function
        reconstruction_model = None
        compression_model = None
        classification_model = None
        model_name = '%s_%s_%s' % (model_class, sharing_ray, grafting_depth)
        # Defining input tensors, one for numerical and one for each categorical
        input_datas = []
        input_datas.append(Input(shape=(self.numerical_features_length,), name='num_input'))
        for i, nom_length in enumerate(self.nominal_features_lengths):
            input_datas.append(Input(shape=(nom_length,), name='nom_input%s' % i, sparse=True))

        # TODO - StackedAutoencoder per-packet anomaly detection.
        if model_class == 'ksae':
            reconstruction_models = []
            compression_models = []
            x = None
            for _ in range(32):
                reconstruction_model, compression_model, _, _ = self.init_model(model_class='kdae')
                if len(compression_model.outputs) > 1:
                    x = concatenate(compression_model.outputs)
                else:
                    x = compression_model.output
                reconstruction_models.append(reconstruction_model)
                compression_models.append(compression_model)

        if model_class == 'kdae' or model_class == 'kc2dae':
            # shared_factors = []
            # if sharing_ray > 0:
            #     shared_factors = reduction_factors[-sharing_ray:] + reduction_factors[::-1][1:sharing_ray + 1]
            # print(shared_factors)
            # encodeds = []
            # decodeds = []
            # xs = []
            # # Variables for variational model
            # z_means = []
            # z_log_var = []
            # # Variable for propagational model
            # propagateds = {}
            #
            # def sampling(args):
            #     """Reparameterization trick by sampling from an isotropic unit Gaussian.
            #     # Arguments
            #         args (tensor): mean and log of variance of Q(z|X)
            #     # Returns
            #         z (tensor): sampled latent vector
            #     """
            #     z_mean, z_log_var = args
            #     batch = K.shape(z_mean)[0]
            #     dim = K.int_shape(z_mean)[1]
            #     # by default, random_normal has mean = 0 and std = 1.0
            #     epsilon = K.random_normal(shape=(batch, dim))
            #     return z_mean + K.exp(0.5 * z_log_var) * epsilon
            #
            # # Building model for numerical features
            # x = input_datas[0]
            # for i in range(np.min([self.depth, grafting_depth])):
            #     output_dim = int(np.ceil(reduction_factors[i] * self.numerical_features_length))
            #     if i < self.depth - 1:
            #         print('Adding Numerical Encoder')
            #         x = Dense(output_dim, activation=num_hidden_activation_function, name='num_enc%s' % i)(x)
            #     if i == self.depth - 1:  # If numerical model reaches the max depth (no grafting)
            #         if not self.variational:
            #             print('Adding Last Numerical Encoder')
            #             x = Dense(output_dim, activation=num_hidden_activation_function, name='num_enc%s' % i)(x)
            #         else:  # Variational last output
            #             latent_dim = int(np.ceil(latent_ratio * self.numerical_features_length))
            #             z_mean = Dense(latent_dim, name='num_z_mean%s' % (i))(x)
            #             z_log_var = Dense(latent_dim, name='num_z_log_var%s' % (i))(x)
            #             x = Lambda(sampling, output_shape=(latent_dim,), name='num_z%s' % (i))([z_mean, z_log_var])
            #         if self.sparse:  # Sparsifying last output
            #             x.W_regularizer = regularizers.l1(1e-4)
            #         encodeds.append(x)
            #     if self.propagational:
            #         if 'num' not in propagateds:
            #             propagateds['num'] = []
            #         propagateds['num'].append(x)
            # graft_in = x
            # graft_out_bk = x
            # graft_out = []
            # # Building model for nominal features
            # for i, nom_length in enumerate(self.nominal_features_lengths):
            #     # Instantiate separated inputs
            #     x = input_datas[i + 1]
            #     for j in range(self.depth - sharing_ray):
            #         if j == grafting_depth:  # Saving layer to be grafted as graft_in
            #             concat_ix = concatenate([graft_in, x])
            #             x = concat_ix
            #         output_dim = int(np.ceil(reduction_factors[j] * nom_length))
            #         if j < self.depth - 1:
            #             print('Adding Nominal %s Encoder' % j)
            #             x = Dense(output_dim, activation=nom_hidden_activation_function, name='nom_enc%s%s' % (i, j))(x)
            #         if j == self.depth - 1:
            #             if not self.variational:
            #                 print('Adding Last Nominal %s Encoder' % j)
            #                 x = Dense(output_dim, activation=nom_hidden_activation_function,
            #                           name='nom_enc%s%s' % (i, j))(x)
            #             else:
            #                 latent_dim = int(np.ceil(latent_ratio * nom_length))
            #                 z_mean = Dense(latent_dim, name='nom_z_mean%s' % (i))(x)
            #                 z_log_var = Dense(latent_dim, name='nom_z_log_var%s' % (i))(x)
            #                 x = Lambda(sampling, output_shape=(latent_dim,), name='nom_z%s' % (i))([z_mean, z_log_var])
            #             if self.sparse:  # Sparsifying last output
            #                 x.W_regularizer = regularizers.l1(1e-4)
            #             encodeds.append(x)
            #         if self.propagational:
            #             if 'nom%s' % i not in propagateds:
            #                 propagateds['nom%s' % i] = []
            #             propagateds['nom%s' % i].append(x)
            #     xs.append(x)
            # if len(xs) > 0:
            #     concat_xs = concatenate(xs)
            #     x = concat_xs
            #     for i, join_factor in enumerate(shared_factors):
            #         shared_length = np.sum(self.nominal_features_lengths)
            #         if grafting_depth < self.depth:
            #             shared_length += self.numerical_features_length
            #         output_dim = int(np.ceil(join_factor * shared_length))
            #         if i < np.floor(len(shared_factors) / 2.):  # Adding encoding shared layers
            #             if i == (grafting_depth - self.depth + sharing_ray):  # Saving graft_in
            #                 concat_ix = concatenate([graft_in, x])
            #                 x = concat_ix
            #             if i != np.floor(len(shared_factors) / 2.) - 1:
            #                 print('Adding Shared Encoder')
            #                 x = Dense(output_dim, activation=hyb_activation_function, name='nom_jenc%s' % (i))(x)
            #             else:
            #                 if not self.variational:
            #                     print('Adding Last Shared Encoder')
            #                     x = Dense(output_dim, activation=hyb_activation_function, name='nom_jenc%s' % (i))(x)
            #                 else:
            #                     latent_dim = int(np.ceil(latent_ratio * shared_length))
            #                     z_mean = Dense(latent_dim, name='nom_z_mean%s' % (i))(x)
            #                     z_log_var = Dense(latent_dim, name='nom_z_log_var%s' % (i))(x)
            #                     x = Lambda(sampling, output_shape=(latent_dim,), name='nom_z%s' % (i))(
            #                         [z_mean, z_log_var])
            #                 if self.sparse:  # Sparsifying middle output
            #                     x.W_regularizer = regularizers.l1(1e-4)
            #                 encodeds.append(x)
            #             if self.propagational:
            #                 if 'hyb' not in propagateds:
            #                     propagateds['hyb'] = []
            #                 propagateds['hyb'].append(x)
            #         else:  # Adding decoding shared layers
            #             print('Adding Shared Decoder')
            #             x = Dense(output_dim, activation=hyb_activation_function,
            #                       name='nom_jdec%s' % (i - sharing_ray - 1))(x)
            #             if self.propagational:
            #                 x = concatenate([x, propagateds['hyb'][-1]])
            #                 x = Dense(output_dim, activation=hyb_activation_function,
            #                           name='nom_pjdec%s' % (i - sharing_ray - 1))(x)
            #                 del propagateds['hyb'][-1]
            #             if i == (-grafting_depth + self.depth + sharing_ray):  # Saving graft_out
            #                 graft_out.append(x)
            #         xs = [x] * len(self.nominal_features_lengths)
            # for i, nom_length in enumerate(self.nominal_features_lengths):
            #     x = xs[i]
            #     for j in reversed(range(self.depth - np.max([0, sharing_ray]))):  # Adding decoder layers
            #         output_dim = int(np.ceil(reduction_factors[j] * nom_length))
            #         print('Adding Nominal %s Decoder' % j)
            #         x = Dense(output_dim, activation=nom_hidden_activation_function,
            #                   name='nom_dec%s%s' % (i, self.depth - j - 1))(x)
            #         if self.propagational:
            #             x = concatenate([x, propagateds['nom%s' % i][-1]])
            #             x = Dense(output_dim, activation=hyb_activation_function,
            #                       name='nom_pdec%s%s' % (i, self.depth - j - 1))(x)
            #             del propagateds['nom%s' % i][-1]
            #         if j == grafting_depth:  # If layer is graft layer, save graft_out
            #             graft_out.append(x)
            #     # Instantiate separated outputs
            #     decodeds.append(
            #         Dense(nom_length, activation=self.nominal_output_activation_functions[i], name='nom_out%s' % (i))(
            #             x))
            # if len(graft_out) > 1:  # More than one graft layer, if graft is on non shared layers
            #     x = concatenate(graft_out)
            #     print('0')
            # elif len(graft_out) > 0:  # Only one graft layer, if graft is on a shared layer
            #     x = graft_out[0]
            #     print('1')
            # else:  # When no graft, we use previously saved numerical output
            #     x = graft_out_bk
            #     print('2')
            # for i in reversed(range(np.min([self.depth, grafting_depth]))):  # Adding decoder layers
            #     output_shape = int(np.ceil(reduction_factors[i] * self.numerical_features_length))
            #     print('Adding Numerical Decoder')
            #     x = Dense(output_shape, activation=self.numerical_output_activation_function,
            #               name='num_dec%s' % (self.depth - i - 1))(x)
            #     if self.propagational:
            #         x = concatenate([x, propagateds['num'][-1]])
            #         x = Dense(output_shape, activation=hyb_activation_function,
            #                   name='num_pdec%s' % (self.depth - i - 1))(x)
            #         del propagateds['num'][-1]
            # decodeds = [Dense(self.numerical_features_length, activation=self.numerical_output_activation_function,
            #                   name='num_out')(x)] + decodeds

            # def LSG_DAE_loss(y_true, y_pred, depth, sharing_ray, grafting_depth, losses):
            #
            #
            #
            #     return 0

            encodeds = []
            decodeds = []
            xs = []

            num_shape = []
            nom_shape = []
            hyb_shape = []

            num_out = []
            nom_out = []
            hyb_out = []

            hybrid = False

            input_shape = input_datas[0]._keras_shape[1]
            input_layer_denses = [
                Dense(input_shape, activation=num_hidden_activation_function, name='num_penc')(input_datas[0])
            ]
            for i, input_data in enumerate(input_datas[1:]):
                input_shape = input_data._keras_shape[1]
                input_layer_denses.append(
                    Dense(input_shape, activation=nom_hidden_activation_function, name='num_penc%s' % i)(input_data)
                )
            # Select first input, that is the numerical
            x = input_datas[0]
            # Numerical encoder deepness depends on grafting depth
            for i in range(grafting_depth):
                input_shape = x._keras_shape[1]
                output_shape = int(np.ceil(input_shape * reduction_coeff))
                num_shape.append(input_shape)
                x = Dense(output_shape, activation=num_hidden_activation_function, name='num_enc%s' % i)(x)
                if i == self.depth - 1:
                    encodeds.append(x)
            # The last numerical tensor should graft on each nominal encoder
            graft_in = x
            for i in range(len(self.nominal_features_lengths)):
                # Select the i+1 input, that in the current nominal
                x = input_datas[i + 1]
                nom_shape.append([])
                # Nominal encoder deepness depends on sharing ray
                for j in range(self.depth - sharing_ray):
                    # Since numerical could graft on each level, we first control this
                    if j == grafting_depth:
                        x = concatenate([graft_in, x])
                    input_shape = x._keras_shape[1]
                    output_shape = int(np.ceil(input_shape * reduction_coeff))
                    nom_shape[-1].append(input_shape)
                    x = Dense(output_shape, activation=nom_hidden_activation_function, name='nom_enc%s%s' % (i, j))(x)
                    if j == self.depth - 1:
                        encodeds.append(x)
                # The last nominal tensor should be the input of shared layers
                xs.append(x)
            x = concatenate(xs)
            # Shared layers depends on sharing ray
            for i in range(self.depth - sharing_ray, self.depth):
                hybrid = True
                if i == grafting_depth:
                    x = concatenate([graft_in, x])
                input_shape = x._keras_shape[1]
                output_shape = int(np.ceil(input_shape * reduction_coeff))
                hyb_shape.append(input_shape)
                x = Dense(output_shape, activation=hyb_activation_function, name='nom_henc%s' % (i))(x)
                if i == self.depth - 1:
                    encodeds.append(x)
            # Decoder
            graft_out = []
            for i in reversed(range(self.depth - sharing_ray, self.depth)):
                output_shape = hyb_shape[i - self.depth]
                x = Dense(output_shape, activation=hyb_activation_function, name='nom_hdec%s' % (i))(x)
                if i == grafting_depth:
                    graft_out.append(x)
                if i == 0:
                    hyb_out.append(x)
                    # decodeds.append(x)
            x_h = x
            for i in range(len(self.nominal_features_lengths)):
                if hybrid:
                    x = x_h
                else:
                    x = xs[i]
                # Nominal decoder deepness depends on sharing ray
                for j in reversed(range(self.depth - sharing_ray)):
                    output_shape = nom_shape[i][j]
                    x = Dense(output_shape, activation=nom_hidden_activation_function, name='nom_dec%s%s' % (i, j))(x)
                    # Since numerical could graft on each level, we first control this
                    if j == grafting_depth:
                        graft_out.append(x)
                    if j == 0:
                        nom_out.append(x)
                        # decodeds.append(x)
            # If graft out has a size gt 1, it derives from multiple nominal layer and needs concatenation
            if len(graft_out) > 1:
                x = concatenate(graft_out)
            # If graft out has a size eq 1, it derives from hybrid layer and needs concatenation
            elif len(graft_out) == 1:
                x = graft_out[0]
            # If graft out is empty, we should take the numerical graft in as input
            else:
                x = graft_in
            # Numerical decoder deepness depends on grafting depth
            for i in reversed(range(grafting_depth)):
                output_shape = num_shape[i]
                x = Dense(output_shape, activation=num_hidden_activation_function, name='num_dec%s' % i)(x)
                if i == 0:
                    num_out.append(x)
                    # decodeds.append(x)

            if len(num_out) > 0:
                decodeds.append(
                    Dense(self.numerical_features_length, activation=self.numerical_output_activation_function,
                          name='num_out')(num_out[0]))
            elif len(nom_out) > 0:
                decodeds.append(
                    Dense(self.numerical_features_length, activation=self.numerical_output_activation_function,
                          name='num_out')(concatenate(nom_out)))
            elif len(hyb_out) > 0:
                decodeds.append(
                    Dense(self.numerical_features_length, activation=self.numerical_output_activation_function,
                          name='num_out')(hyb_out[0]))
            for i, (nominal_features_length, nominal_output_activation_function) in enumerate(
                    zip(self.nominal_features_lengths, self.nominal_output_activation_functions)
            ):
                if len(nom_out) > 0:
                    decodeds.append(
                        Dense(nominal_features_length, activation=nominal_output_activation_function,
                              name='nom_out%s' % i)(nom_out[i])
                    )
                elif len(hyb_out) > 0:
                    decodeds.append(
                        Dense(nominal_features_length, activation=nominal_output_activation_function,
                              name='nom_out%s' % i)(hyb_out[0])
                    )

            # For fitting
            reconstruction_model = Model(input_datas, decodeds)
            # For feature extraction
            compression_model = Model(input_datas, encodeds)
            plot_model(reconstruction_model,
                       to_file=self.plot_file + '_reconstruction_jr%s_il%s' % (sharing_ray, grafting_depth) + '.png',
                       show_shapes=True, show_layer_names=True)
            # plot_model(reconstruction_model,
            #            to_file='./reconstruction_jr%s_il%s' % (sharing_ray, grafting_depth) + '.png', show_shapes=True,
            #            show_layer_names=True)
            with open(self.summary_file + '_reconstruction_jr%s_il%s' % (sharing_ray, grafting_depth) + '.dat',
                      'w') as f:
                reconstruction_model.summary(print_fn=lambda x: f.write(x + '\n'))
            # reconstruction_model.add_loss(LSG_DAE_loss(input_datas, decodeds, self.depth, sharing_ray, grafting_depth, [self.numerical_loss_function] + self.nominal_loss_functions))
            reconstruction_model.compile(optimizer=Adadelta(lr=1.),
                                         loss=[self.numerical_loss_function] + self.nominal_loss_functions,
                                         loss_weights=[self.numerical_weight] + self.nominal_weights)
            if classify:
                reconstruction_model, compression_model, _, _ = self.init_model(model_class, sharing_ray,
                                                                                grafting_depth)
                if len(compression_model.outputs) > 1:
                    x = concatenate(compression_model.outputs)
                else:
                    x = compression_model.output
                initial_size = int(x.shape[1])
                final_size = num_classes
                step = (initial_size - final_size) / self.depth
                mid_sizes = [int(initial_size - step * i) for i in range(1, self.depth)]
                for i, mid_size in enumerate(mid_sizes):
                    x = Dense(mid_size, activation=self.hidden_activation_function, name='clf_hid%s' % i)(x)
                x = Dropout(.1)(x)
                x = Dense(final_size, activation='softmax', name='clf_out')(x)
                classification_model = Model(compression_model.inputs, x)
                plot_model(classification_model, to_file=self.plot_file + '_classification_jr%s_il%s' % (
                    sharing_ray, grafting_depth) + '.png', show_shapes=True, show_layer_names=True)
                with open(self.summary_file + '_classification_jr%s_il%s' % (sharing_ray, grafting_depth) + '.dat',
                          'w') as f:
                    classification_model.summary(print_fn=lambda x: f.write(x + '\n'))
                classification_model.compile(optimizer=Adadelta(lr=1.), loss='categorical_crossentropy')
        if model_class == 'kmlp':
            final_size = num_classes
            mid_size = 100
            if len(input_datas) == 1:
                x = input_datas[0]
            else:
                x = concatenate(input_datas)
            for i in range(self.depth):
                x = Dense(mid_size, activation=self.hidden_activation_function, name='clf_hid%s' % i)(x)
            x = Dropout(.1)(x)
            x = Dense(final_size, activation='softmax', name='clf_out')(x)
            classification_model = Model(input_datas, x)
            plot_model(classification_model, to_file=self.plot_file + '_d_%s_classification' % (self.depth) + '.png',
                       show_shapes=True, show_layer_names=True)
            with open(self.summary_file + '_d_%s_classification' % (self.depth) + '.dat', 'w') as f:
                classification_model.summary(print_fn=lambda x: f.write(x + '\n'))
            classification_model.compile(optimizer=Adadelta(lr=1.), loss='categorical_crossentropy')
        # TODO: convolutional layer for mlp to flatten matricial input of weights from NeuralWeightsDecisor
        if model_class == 'kcmlp':
            final_size = num_classes
            mid_size = 100
            if len(input_datas) == 1:
                x = input_datas[0]
            else:
                x = concatenate(input_datas)
            x = Conv2D()
            for i in range(self.depth):
                x = Dense(mid_size, activation=self.hidden_activation_function, name='clf_hid%s' % i)(x)
            x = Dropout(.1)(x)
            x = Dense(final_size, activation='softmax', name='clf_out')(x)
            classification_model = Model(input_datas, x)
            plot_model(classification_model, to_file=self.plot_file + '_d_%s_classification' % (self.depth) + '.png',
                       show_shapes=True, show_layer_names=True)
            with open(self.summary_file + '_d_%s_classification' % (self.depth) + '.dat', 'w') as f:
                classification_model.summary(print_fn=lambda x: f.write(x + '\n'))
            classification_model.compile(optimizer=Adadelta(lr=1.), loss='categorical_crossentropy')
        return reconstruction_model, compression_model, classification_model, model_name

    def predict(self, X, y=None):

        core.utils.preprocessing.ohe(X, self.nominal_features_index, self.nominal_encoder)
        self.numerical_features_length, self.nominal_features_lengths = core.utils.preprocessing.get_num_nom_lengths(X)

        # X, nominal_features_index, numerical_features_index = self.expand_onehot_features(X)
        nom_X, num_X = self.split_nom_num_features(X)
        scaled_num_X = self.scaler.transform(num_X)
        if self.classify:
            pred = self.classification_model_.predict([scaled_num_X] + nom_X)
            self.proba = copy(pred)
            for i, sp in enumerate(pred):
                sp_max = np.max(sp)
                n_max = len(pred[pred == sp_max])
                pred[i] = [1 if p == sp_max else 0 for p in sp]
                indexes = np.where(sp == 1)[0]
                if len(indexes) > 1:
                    rand_sp = np.random.choice(indexes)
                    pred[i] = [1 if j == rand_sp else 0 for j in range(len(sp))]
            pred = self.label_encoder.inverse_transform(pred)
        else:
            reconstr_X = self.reconstruction_model_.predict([scaled_num_X] + nom_X)
            losses = []
            losses.append(K.eval(globals()[self.numerical_loss_function](scaled_num_X, reconstr_X[0])))
            for nts, rnts, nominal_loss_function in zip(nom_X, reconstr_X[1:], self.nominal_loss_functions):
                losses.append(K.eval(globals()[nominal_loss_function](nts.toarray(), K.constant(rnts))))
            # first losses refers to numerical features, so we weighted them
            # num_weight = [ self.numerical_features_length ]
            # nom_weights = [1] * len(self.nominal_features_lengths)
            # loss = np.average(losses, axis = 0, weights = num_weight + nom_weights)
            # Weighted sum of losses for each sample
            loss = np.dot(np.array(losses).T, [self.numerical_weight] + self.nominal_weights)
            # loss = [ (l-self._loss_min)/(self._loss_max-self._loss_min) + eps for l in loss ]
            pred = np.asarray([
                -1 if l > self.normal_loss_
                else 1 for l in loss
            ])
            self.proba = np.asarray(loss)
            self.proba = np.reshape(self.proba, self.proba.shape[0])
        pred = np.reshape(pred, pred.shape[0])
        return pred

    def predict_proba(self, X):

        if self.proba is None:
            self.predict(X)
        return self.proba

    def score(self, X, y=None):

        return 1 / np.mean(self.predict_proba(X))

    def set_oracle(self, oracle):

        self.oracle = oracle

    def sklearn2keras(self, features, labels=None):

        pass

    def expand_onehot_features(self, set):

        # nominal_features_index = [0,1,2,5]
        nominal_features_index = [i for i, v in enumerate(set[0, :]) if isinstance(v, np.ndarray)]
        nominal_features_lengths = [len(set[0, i]) for i in nominal_features_index]

        nom_len = [0]
        nom_base = [0]
        for i, nfi in enumerate(nominal_features_index):
            nom_base.append(nfi + nom_len[-1])
            nom_len.append(nom_len[-1] + self.nominal_features_lengths[i] - 1)
            set = np.c_[
                set[:, :nom_base[-1]], np.asarray(list(zip(*set[:, nom_base[-1]]))).T, set[:, nom_base[-1] + 1:]]

        nom_index = nom_base[1:]
        nom_length = self.nominal_features_lengths
        features_number = set.shape[1]

        nominal_features_index = []
        caught_features_index = []
        for nom_i, nom_l in zip(nom_index, nom_length):
            nominal_features_index.append([i for i in range(nom_i, nom_i + nom_l)])
            caught_features_index += range(nom_i, nom_i + nom_l)

        numerical_features_index = [i for i in range(features_number) if i not in caught_features_index]

        return np.asarray(set, dtype=float), \
               np.asarray(nominal_features_index, dtype=object), \
               np.asarray(numerical_features_index, dtype=int)

    def split_nom_num_features(self, set):
        nom_index = []
        num_index = []

        for i, v in enumerate(set[0, :]):
            # if isinstance(v, np.ndarray):
            if issparse(v):
                nom_index.append(i)
            else:
                num_index.append(i)

        nom_set = set[:, nom_index]
        num_set = set[:, num_index]

        temp_set = []
        for i in range(nom_set.shape[1]):
            temp_set.append(nom_set[:, i])

        # Fare un nom_set list di 4 ndarray ognuno con tutte le colonne dentro
        nom_set = [csr_matrix([sp[0].toarray()[0] for sp in np.vstack(v)]) for v in temp_set]
        # nom_set = [ csr_matrix(v) for v in temp_set ]

        return nom_set, np.asarray(num_set, dtype=float)
