from sklearn.model_selection import GridSearchCV
import random
from collections import OrderedDict
import itertools
import math
import scipy.special as spec
from time import time
from copy import deepcopy
import numpy as np
from sklearn.model_selection import StratifiedKFold
from concurrent.futures import ThreadPoolExecutor


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
                 verbose=0, pre_dispatch='2*n_jobs', error_score=nan, return_train_score=False):
        super(Sklearn_GridSearchCV, self).__init__(estimator, param_grid, scoring=scoring, n_jobs=n_jobs,
                                                   iid=iid, refit=refit, cv=cv,
                                                   verbose=verbose, pre_dispatch=pre_dispatch,
                                                   error_score=error_score,
                                                   return_train_score=return_train_score)


class Custom_GeneticAlgorithm(object):
    def __init__(self, estimator, param_grid, generations=50, population_size=20, min_delta=1e-4, mutation_proba=.1,
                 n_jobs=None, verbose=1):
        random.seed(0)
        self.estimator = estimator
        self.generations = generations
        self.population_size = population_size
        if isinstance(param_grid, list):
            param_grid = merge_dicts(param_grid)
        self.param_grid = OrderedDict(param_grid.items())
        self.min_delta = min_delta
        self.mutation_proba = mutation_proba
        self.n_jobs = None
        self.verbose = verbose

        self.parents_n = math.ceil(population_size * .25)
        print('parents_n', self.parents_n)
        self.lucky_unpaireds_n = math.ceil((population_size - self.parents_n) * .1)
        print('lucky_unpaireds_n', self.lucky_unpaireds_n)
        offsprings_n = spec.comb(self.parents_n, 2) * 2
        print('offsprings_n', offsprings_n)
        population_control_factor = (population_size - self.lucky_unpaireds_n - self.parents_n) / (
                (self.parents_n - 1) * self.parents_n)
        print('population_control_factor', population_control_factor)
        self.offsprings_planned_n = math.ceil(offsprings_n * population_control_factor)
        print('offsprings_planned_n', self.offsprings_planned_n)

        self.population = [None] * population_size
        self.old_max_fitness_value = 0
        self.killed = list()

        self.memoized_score_estimators_dict = dict()

        self.best_estimator_ = None
        self.cv_results_ = dict()

    def verbose_print(self, *args):
        if self.verbose:
            return print(args)

    def fit(self, X, y):
        self.verbose_print('Initialize population')
        self.init_population()
        generation_cnt = 0
        while True:
            self.verbose_print('Generation %s' % generation_cnt)
            self.verbose_print('Population size: %s' % self.population_size)
            fitness_values = list()
            self.verbose_print('Compute fitness functions')
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = list()
                for i, chromosome in enumerate(self.population):
                    self.verbose_print('Model %s:' % i, {attribute: gene for attribute, gene in zip(
                        self.param_grid.keys(), chromosome)})
                    attributes = {attribute: gene for attribute, gene in zip(
                        self.param_grid.keys(), chromosome)}
                    futures.append(executor.submit(self.fitness_function, attributes, X, y))
                for i, future in enumerate(futures):
                    fitness_values.append(future.result())
                    self.verbose_print('Fitness value Model %s:' % i, fitness_values[-1])
            parents_index = self.parents_selection(fitness_values, self.parents_n)
            self.curr_max_fitness_value = np.max(fitness_values)
            self.verbose_print('Best fitness value of Generation %s:' % generation_cnt, self.curr_max_fitness_value)
            is_not_better = (
                                    self.curr_max_fitness_value - self.old_max_fitness_value) < self.min_delta
            is_last_gen = generation_cnt >= self.generations
            if is_not_better or is_last_gen:
                self.verbose_print('End of evolution')
                estimator = self.estimator(**{attribute: gene for attribute, gene in zip(
                    self.param_grid, self.population[parents_index[0]])})
                estimator.fit(X, y)
                self.best_estimator_ = estimator
                return self.best_estimator_
            self.old_max_fitness_value = self.curr_max_fitness_value
            self.verbose_print('Lucky unpaired selection')
            lucky_unpaireds = self.lucky_umpaireds_selection(parents_index)
            self.verbose_print('Application of mutation to unpaireds')
            for lucky_unpaired in lucky_unpaireds:
                lucky_unpaired = self.mutation(lucky_unpaired)
            self.verbose_print('Compute mating set')
            mating_set = self.mating_pool(parents_index)
            offsprings = list()
            self.verbose_print('Application of crossover')
            for mating in mating_set:
                offsprings.extend(self.crossover(mating))
            self.verbose_print('Application of mutation to offsprings')
            for offspring in offsprings:
                offspring = self.mutation(offspring)
            offsprings = self.population_control(offsprings)
            self.verbose_print('Population renewal')
            parents = [self.population[i] for i in parents_index]
            self.population = parents + lucky_unpaireds + offsprings
            generation_cnt += 1

    def init_population(self):
        for i in range(self.population_size):
            while True:
                chromosome = list()
                for attribute in self.param_grid:
                    gene_value = random.choice(self.param_grid[attribute])
                    chromosome.append(gene_value)
                if chromosome not in self.population:
                    self.population[i] = chromosome
                    break

    def fitness_function(self, attributes, X, y):
        # return random.random()
        if str(attributes) in self.memoized_score_estimators_dict:
            return self.memoized_score_estimators_dict[str(attributes)]
        else:
            # t = time()
            estimator = self.estimator(**attributes)
            skf = StratifiedKFold(n_splits=3)
            scores = []
            for train_index, validation_index in skf.split(X, y):
                estimator_under_test = deepcopy(estimator)
                estimator_under_test.fit(X[train_index], y[train_index])
                scores.append(estimator_under_test.score(X[validation_index], y[validation_index]))
            score = np.mean(scores)
            # t = time() - t
            # score = (math.exp(score) + math.exp(1 / t)) / 2
            self.memoized_score_estimators_dict[str(attributes)] = score
            self.cv_results_.setdefault('params', []).append(str(attributes))
            self.cv_results_.setdefault('rank_test_score', []).append(score)
        return score

    def parents_selection(self, fitness_values, parents_n):
        parents_index = list(reversed(np.argsort(fitness_values)))[:parents_n]
        return parents_index

    def mating_pool(self, parents_index):
        mating_set = list(set([tuple(sorted(c)) for c in itertools.combinations(
            parents_index, 2)]))
        return mating_set

    def crossover(self, parents_index, k=3):
        # while True:
        offsprings = [None, None]
        parent_0 = self.population[parents_index[0]]
        parent_1 = self.population[parents_index[1]]
        points = list(sorted([random.randint(1, len(parent_0)) for _ in range(k)]))
        for i, point in enumerate(points + [points[-1]]):
            if i == 0:
                offsprings[0] = parent_0[:point]
                offsprings[1] = parent_1[:point]
            elif i % 2 == 0:
                if i == k:
                    offsprings[0] += parent_0[point:]
                    offsprings[1] += parent_1[point:]
                else:
                    offsprings[0] += parent_0[points[i - 1]:point]
                    offsprings[1] += parent_1[points[i - 1]:point]
            else:
                if i == k:
                    offsprings[0] += parent_1[point:]
                    offsprings[1] += parent_0[point:]
                else:
                    offsprings[0] += parent_1[points[i - 1]:point]
                    offsprings[1] += parent_0[points[i - 1]:point]
        # if offsprings[0] not in self.population and offsprings[1] not in self.population:
        return offsprings

    def mutation(self, chromosome):
        # Continuiamo a mutare il cromosoma oppure partiamo ogni volta dallo stesso cromosoma? per ora la prima
        # while True:
        mutation_n = random.randint(1, len(chromosome))
        for _ in range(mutation_n):
            mutate = random.random() <= self.mutation_proba
            if mutate:
                mutation_p = random.randint(0, len(chromosome) - 1)
                mutated_value = random.choice(self.param_grid[list(self.param_grid.keys())[mutation_p]])
                chromosome[mutation_p] = mutated_value
            # if chromosome not in self.population:
        return chromosome

    def lucky_umpaireds_selection(self, parents_index):
        unpaireds_index = [i for i in range(self.population_size) if i not in parents_index]
        lucky_unpaireds_index = random.sample(unpaireds_index, self.lucky_unpaireds_n)
        lucky_unpaireds = [self.population[i] for i in lucky_unpaireds_index]
        # Save killed
        self.killed += [self.population[i] for i in unpaireds_index if i not in lucky_unpaireds_index]
        return lucky_unpaireds

    def population_control(self, offsprings):
        offsprings = [list(offspring) for offspring in set([tuple(offspring) for offspring in offsprings])]
        assert len(offsprings) >= self.offsprings_planned_n, 'Warning: not enough unique offspring'
        offsprings = offsprings[:self.offsprings_planned_n]
        return offsprings
