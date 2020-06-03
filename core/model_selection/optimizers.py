import os
import itertools
import math
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

import numpy as np
import scipy.special as spec
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


def merge_dicts(dicts):
    merged_dict = dict()
    keys = list(set([key for d in dicts for key in list(d.keys())]))
    for key in keys:
        merged_dict.setdefault(key, [])
        for d in dicts:
            value = d.get(key, [])
            if isinstance(value, list):
                merged_dict[key] += value
            else:
                merged_dict[key] += [value]
    return merged_dict


class Sklearn_GridSearchCV(GridSearchCV):
    def __init__(self, estimator, param_grid, scoring=None, n_jobs=None, iid='deprecated', refit=True, cv=None,
                 verbose=0, pre_dispatch='2*n_jobs', error_score=None, return_train_score=False):
        super(Sklearn_GridSearchCV, self).__init__(estimator, param_grid, scoring=scoring, n_jobs=n_jobs,
                                                   iid=iid, refit=refit, cv=cv,
                                                   verbose=verbose, pre_dispatch=pre_dispatch,
                                                   error_score=error_score,
                                                   return_train_score=return_train_score)


class Custom_GeneticAlgorithm(object):
    def __init__(self, estimator, param_grid, generations=50, population_size=20, min_delta=1e-4, mutation_proba=.1,
                 common_param=dict(), modality='max', n_jobs=1, verbose=1, seed=0):
        """
        :param estimator: class of the estimator to optimize.
        :param param_grid: dictionary as {hyperparameter_name: [value_0, ...], ...}, estimator's hyperparameters to
        optimize.
        :param generations: integer, number of generation to wait until the end.
        :param population_size: integer, size of the population at each generation.
        :param min_delta: float, minimum tolerable score increment. Genetic Algorithm stops if improvement is under
        min_delta.
        :param mutation_proba: float in (0, 1], probability a mutation occurs.
        :param common_param: dictionary as {hyperparameter_name: value, ...}, fixed estimator's hyperparameters.
        :param modality: string in {'max', 'min'}, optimization target, default is 'max'.
        :param n_jobs: integer, number of process for optimization parallelization, default is 1.
        :param verbose: integer in {0, 1}.
        :param seed: integer, np.random seed, default is 0.
        """
        np.random.seed(seed)
        self.estimator = estimator
        self.generations = generations
        self.population_size = population_size
        if isinstance(param_grid, list):
            param_grid = merge_dicts(param_grid)
        self.param_grid = OrderedDict(param_grid.items())
        self.min_delta = min_delta
        self.mutation_proba = mutation_proba
        self.common_param = common_param
        self.modality = modality
        self.n_jobs = n_jobs
        self.verbose = verbose

        parents_ratio = .25
        bachelors_ratio = .075
        n_offsprings_per_parents = 2

        # Parents are the 25% of the entire population.
        self.n_parents = math.ceil(population_size * parents_ratio)
        print('n_parents', self.n_parents)
        # Lucky bachelors are the 7.5% of the entire population.
        self.n_lucky_bachelors = math.ceil(population_size * bachelors_ratio)
        print('n_lucky_bachelors', self.n_lucky_bachelors)
        # Total number of offsprings is obtained by combining 2 diverse parents per time.
        # Two parents generate n_offsprings_per_parents.
        n_couples = spec.comb(self.n_parents, 2)
        print('n_couples', n_couples)
        n_offsprings = n_couples * n_offsprings_per_parents
        print('n_offsprings', n_offsprings)
        # Planned offsprings are the rest of the population (i.e. w/o parents and bachelors)
        self.n_planned_offsprings = population_size - self.n_lucky_bachelors - self.n_parents
        print('n_planned_offsprings', self.n_planned_offsprings)
        # Population control factor is the percentage of selected offsprings, the rest down at Taigeto.
        population_control_factor = self.n_planned_offsprings / n_offsprings
        print('population_control_factor', population_control_factor)

        assert n_offsprings >= self.n_planned_offsprings, 'Error: not enough offsprings.'

        self.population = list()
        self.old_best_fitness_value = 0
        # List of discarded models
        self.dead = list()

        # Variable to handle already evaluated models, here are stored hyperparameters configuration and corresponding
        # score
        self.memoized_score_estimators_dict = dict()

        self.best_estimator_ = None
        self.cv_results_ = dict()

    def vprint(self, *args):
        if self.verbose:
            return print(args)

    def fit(self, X, y):
        """
        :param X: matrix, features in the form of (nsamples, nfeatures)
        :param y: array, labels (nsamples, )
        :return: best_estimator fitted on X, y
        TODO: manage X and y as shared variables between processes to avoid RAM saturation
        TODO: leverage workers attribute of keras.models.Model.fit method instead of current management
        """
        self.vprint('Initialize population')
        self.init_population()
        generation_cnt = 0
        while True:
            # Evaluation of current generation
            self.vprint('Generation %s' % generation_cnt)
            self.vprint('Population size: %s' % self.population_size)
            fitness_values = list()
            self.vprint('Compute fitness function')
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = list()
                for i, chromosome in enumerate(self.population):
                    self.vprint('Model %s:' % i, {attribute: gene for attribute, gene in zip(
                        self.param_grid.keys(), chromosome)})
                    attributes = {attribute: gene for attribute, gene in zip(
                        self.param_grid.keys(), chromosome)}
                    #futures.append(executor.submit(self.fitness_function, attributes, X, y))
                    fitness_values.append(self.fitness_function(attributes, X, y))
                #for i, future in enumerate(futures):
                    #fitness_values.append(future.result())
                    self.vprint('Fitness value Model %s:' % i, fitness_values[-1])
            # Selecting the best n_parents parents
            parents_indexes = self.parents_selection(fitness_values)
            # Updating current best fitness value
            self.curr_best_fitness_value = np.max(fitness_values) if self.modality == 'max' else np.min(fitness_values)
            generation_cnt += 1
            self.vprint('Best fitness value of Generation %s:' % generation_cnt, self.curr_best_fitness_value)
            gain = (self.curr_best_fitness_value - self.old_best_fitness_value)
            is_better = gain >= self.min_delta if self.modality == 'max' else gain <= self.min_delta
            is_last_gen = generation_cnt == self.generations
            if not is_better or is_last_gen:
                self.vprint('End of evolution')
                estimator = self.estimator(**{attribute: gene for attribute, gene in zip(
                    self.param_grid, self.population[parents_indexes[0]])})
                estimator.fit(X, y)
                self.best_estimator_ = estimator
                return self.best_estimator_
            # Generation of next generation
            self.old_best_fitness_value = self.curr_best_fitness_value
            self.vprint('Lucky bachelor selection')
            lucky_bachelors = self.lucky_bachelors_selection(parents_indexes)
            self.vprint('Application of mutation to bachelors')
            for lucky_bachelor in lucky_bachelors:
                self.mutation(lucky_bachelor)
            self.vprint('Compute mating set')
            mating_set = self.mating_pool(parents_indexes)
            offsprings = list()
            self.vprint('Application of crossover to generate offsprings.')
            for mating in mating_set:
                offsprings.extend(self.crossover(mating))
            self.vprint('Application of mutation to offsprings')
            for offspring in offsprings:
                self.mutation(offspring)
            # Check in enough unique offsprings (>=self.n_planned_offsprings) are generated.
            while not self.check_offsprings(offsprings):
                self.vprint('Re-application of mutation to offsprings')
                for offspring in offsprings:
                    self.mutation(offspring)
            offsprings = self.population_control(offsprings)
            self.vprint('Population renewal')
            parents = [self.population[i] for i in parents_indexes]
            self.population = parents + lucky_bachelors + offsprings

    def init_population(self):
        """
        Function initializes the population to an unique set of chromosomes.  
        """
        for _ in range(self.population_size):
            chromosome = self.generate_chromosome()
            while chromosome in self.population:
                # We avoid to generate the same chromosome twice
                chromosome = self.generate_chromosome()
            self.population.append(chromosome)

    def generate_chromosome(self):
        """
        Function generates a chromosome
        """
        chromosome = list()
        for attribute in self.param_grid:
            gene_values = self.param_grid[attribute]
            gene_value_index = np.random.choice(range(len(gene_values)))
            chromosome.append(gene_values[gene_value_index])
        return chromosome

    def fitness_function(self, attributes, X, y):
        print(os.getpid(), 'STARTED')
        # TODO: implement a training_time dependent fitness_function
        # We add common_parameters to current configuration of attributes
        attributes.update(self.common_param)
        # If set of attributes is already evaluated, return the associated score
        if str(attributes) in self.memoized_score_estimators_dict:
            return self.memoized_score_estimators_dict[str(attributes)]
        # t = time()
        # Instantiate the estimator
        # estimator = self.estimator(**attributes)
        skf = StratifiedKFold(n_splits=3)
        scores = []
        k=0
        for train_index, validation_index in skf.split(X, y):
            estimator_under_test = self.estimator(**attributes)
            k+=1
            #print(os.getpid(), ':', k)
            #estimator_under_test = deepcopy(estimator)
            estimator_under_test.fit(X[train_index], y[train_index])
            scores.append(estimator_under_test.score(X[validation_index], y[validation_index]))
        # Compute the mean score over three fold
        score = np.mean(scores)
        # t = time() - t
        # score = (math.exp(score) + math.exp(1 / t)) / 2
        self.memoized_score_estimators_dict[str(attributes)] = score
        self.cv_results_.setdefault('params', []).append(str(attributes))
        self.cv_results_.setdefault('rank_test_score', []).append(score)
        print(os.getpid(), ':', 'ENDED')
        return score

    def parents_selection(self, fitness_values):
        # Best n_parents are selected by fitness_values
        parents_indexes = list(reversed(np.argsort(fitness_values)))[:self.n_parents]
        return parents_indexes

    def mating_pool(self, parents_indexes):
        # Couple of parents_indexes forming a mating_set
        mating_set = list(set([tuple(sorted(c)) for c in itertools.combinations(
            parents_indexes, 2)]))
        return mating_set

    def crossover(self, parents_indexes, k=3):
        # Crossover between two parents, k is the number of crossover points
        # To speedup convergence, generation of already seen chromosomes is not handled
        """
        Example:
            k =           3
            points =      [   1,    3, 4]
            crossover         V     V  V
            parent_0 =    [0, 1, 2, 3, 4, 5]
            parent_1 =    [5, 4, 3, 2, 1, 0]
            offspring_0 = [0, 4, 3, 3, 1, 0]
            offspring_1 = [5, 1, 2, 2, 4, 5]
        """
        # while True:
        offsprings = [None, None]
        parent_0 = self.population[parents_indexes[0]]
        parent_1 = self.population[parents_indexes[1]]
        points = list(sorted(np.random.choice(range(len(parent_0)), size=k, replace=False)))
        offsprings[0] = parent_0[:points[0]]
        offsprings[1] = parent_1[:points[0]]
        for i, point in enumerate(points[:-1]):
            offsprings[0] += parent_1[point:points[i + 1]]
            offsprings[1] += parent_0[point:points[i + 1]]
            # Swapping parents
            parent_0, parent_1 = parent_1, parent_0
        offsprings[0] += parent_1[points[-1]:]
        offsprings[1] += parent_0[points[-1]:]
        # if offsprings[0] not in self.population and offsprings[1] not in self.population:
        return offsprings

    def mutation(self, chromosome):
        """
        Actually chromosome is cumulatively mutated, i.e. chromosome(t_i) = mutation(chromosome(t_i+1))
        To speedup convergence, we does not assure that mutated chromosomes are unique or never seen on previous
        generations.
        Number of mutating genes is selected randomly.
        TODO: assign a score for each gene based on score of corresponding chromosomes. The score of the gene could
         be useful to weigth the mutation.
        """
        # while True:
        n_mutations = np.random.randint(1, len(chromosome) + 1)
        for _ in range(n_mutations):
            mutate = np.random.np.random() <= self.mutation_proba
            if mutate:
                mutant_gene_index = np.random.randint(0, len(chromosome) - 1)
                gene_values = self.param_grid[list(self.param_grid.keys())[mutant_gene_index]]
                mutated_value_index = np.random.choice(range(len(gene_values)))
                chromosome[mutant_gene_index] = gene_mutants[mutated_value_index]
            # if chromosome not in self.population:
        return chromosome

    def lucky_bachelors_selection(self, parents_indexes):
        bachelors_index = [i for i in range(self.population_size) if i not in parents_indexes]
        lucky_bachelors_index = np.random.choice(bachelors_index, size=self.n_lucky_bachelors, replace=False)
        lucky_bachelors = [self.population[i] for i in lucky_bachelors_index]
        # Save dead
        self.dead += [self.population[i] for i in bachelors_index if i not in lucky_bachelors_index]
        return lucky_bachelors

    def check_offsprings(self, offsprings):
        # Offsprings may not to be unique, due to speedup convergence. This function check if they are enough.
        unique_offsprings = set([tuple(offspring) for offspring in offsprings])
        return len(unique_offsprings) >= self.n_planned_offsprings

    def population_control(self, offsprings):
        """
        Selects randomly n_planned_offsprings within offsprings
        """
        n_offsprings = len(offsprings)
        selected_offspring_index = np.random.choice(range(n_offsprings), size=self.n_planned_offsprings,
                                                    replace=False)
        offsprings = [offsprings[i] for i in selected_offspring_index]
        return offsprings


from keras.models import Model
from keras.layers import *


class Custom_BaseEstimator(BaseEstimator, ClassifierMixin):
    def build_estimator(self, **params: dict) -> Model:
        pass

    def fit(self, X, y):
        pass

    def score(self, X, y, sample_weight=None):
        pass


class MLP(Custom_BaseEstimator):
    def __init__(self, input_shape, output_shape, core_layer=Dense, n_layers=3, n_units=32, dropout_rate=.1,
                 activations='relu', output_activation='softmax',
                 optimizer='rmsprop', loss=None, metrics=None, loss_weights=None, weighted_metrics=None,
                 epochs=100, batch_size=32, callbacks=None, shuffle=True):
        # Common parameters
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.loss = loss
        self.metrics = metrics
        self.loss_weights = loss_weights
        self.weighted_metrics = weighted_metrics
        # Macro-architecture parameters (e.g. type of Net)
        # Micro-architecture parameters
        self.core_layer = core_layer
        self.n_layers = n_layers
        self.n_units = n_units if isinstance(n_units, list) else [n_units] * n_layers
        self.dropout_rate = dropout_rate
        self.activations = activations if isinstance(activations, list) else [activations] * n_layers
        self.output_activation = output_activation
        # Hyperparameters
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.callbacks = callbacks

        self.estimator = self.build_estimator()

    def build_estimator(self):
        input_layer = Input(self.input_shape)
        x = self.core_layer(units=self.n_units[0], activation=self.activations[0])(input_layer)
        for l in range(1, self.n_layers):
            x = self.core_layer(units=self.n_units[l], activation=self.activations[l])(x)
        d = Dropout(self.dropout_rate)(x)
        output_layer = Dense(self.output_shape, activation=self.output_activation)(d)

        model = Model(input_layer, output_layer)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics, loss_weights=self.loss_weights,
                      weighted_metrics=self.weighted_metrics)
        return model

    def fit(self, X, y):
        #print(os.getpid(), ': FIT')
        return self.estimator.fit(X, y, batch_size=self.batch_size, epochs=self.epochs, callbacks=self.callbacks, verbose=2)

    def score(self, X, y, sample_weight=None):
        return self.estimator.evaluate(X, y, sample_weight=sample_weight)
