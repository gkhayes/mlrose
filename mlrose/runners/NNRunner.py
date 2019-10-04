try:
    import mlrose
except:
    import sys

    sys.path.append("..")
    import mlrose

import mlrose.algorithms as mla

from mlrose.runners._RunnerBase import _RunnerBase
from mlrose.neural import RunnerNN
import sklearn.metrics as skmt
import sklearn.model_selection as skms

"""
Example usage:

    experiment_name = 'example_experiment'
    problem = TSPGenerator.generate(seed=SEED, number_of_cities=22)

    sa = NNRunner(problem=problem,
                  experiment_name=experiment_name,
                  output_directory=OUTPUT_DIRECTORY,
                  seed=SEED,
                  iteration_list=2 ** np.arange(14),
                  max_attempts=5000,
                  temperature_list=[1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000])

    # the two data frames will contain the results
    df_run_stats, df_run_curves = sa.run()                  
"""


class NNRunner(_RunnerBase):

    @classmethod
    def runner_name(cls):
        return 'nn'

    def __init__(self, x_train, y_train, x_test, y_test,
                 experiment_name, seed, iteration_list,
                 hidden_nodes_set, activation_set, learning_rates, cv=5,
                 algorithm_set=None, algorithm_params=None,
                 bias=True, early_stopping=False, clip_max=1e+10,
                 max_attempts=500, generate_curves=True, **kwargs):
        super().__init__(problem=None, experiment_name=experiment_name, seed=seed, iteration_list=iteration_list,
                         max_attempts=max_attempts, generate_curves=generate_curves,
                         **kwargs)
        self.hidden_nodes_set = hidden_nodes_set
        self.activation_set = activation_set
        self.learning_rates = learning_rates
        self.algorithm_set = algorithm_set if algorithm_set is not None else [mla.simulated_annealing,
                                                                              mla.genetic_alg,
                                                                              mla.random_hill_climb,
                                                                              mla.mimic]
        # algorithm grid-search params
        self.algorithm_params = algorithm_params
        self.bias = bias
        self.early_stopping = early_stopping
        self.clip_max = clip_max
        self.cv = cv

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def run(self):
        nn = RunnerNN(runner=self,
                      cv_=self.cv,
                      seed_=self.seed,
                      callback_function_=self._save_state)

        # nn grid-search params
        parameters = {
            'hidden_nodes': self.hidden_nodes_set,
            'activation': self.activation_set,
            'learning_rate': self.learning_rates
        }

        sr = self._perform_grid_search(nn, parameters)
        pass

    def _invoke_algorithm(self, algorithm, i, total_args, user_info):
        algorithm(problem=self.problem,
                  curve=self.generate_curves,
                  random_state=self.seed,
                  state_fitness_callback=self._save_state,
                  callback_user_info=user_info,
                  **total_args)

    def run_experiment_(self, algorithm, save_data=True, **kwargs):
        params = kwargs.copy()
        if self.algorithm_params is not None:
            params.update(self.algorithm_params)
        super().run_experiment_(algorithm, save_data, **kwargs)

    def _perform_grid_search(self, classifier, parameters):

        scorer = skmt.make_scorer(skmt.balanced_accuracy_score)
        search_results = skms.GridSearchCV(classifier,
                                           parameters,
                                           cv=self.cv,
                                           scoring=scorer,
                                           # n_jobs=1,
                                           return_train_score=True,
                                           verbose=True)
        search_results.fit(self.x_train, self.y_train)
        return search_results  # , classifier2
