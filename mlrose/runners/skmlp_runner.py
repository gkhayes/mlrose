from sklearn.base import BaseEstimator

import mlrose.neural.activation as act
from sklearn.neural_network import MLPClassifier

from mlrose.decorators import short_name
from mlrose.runners._nn_runner_base import _NNRunnerBase


@short_name('skmlp')
class SKMLPRunner(_NNRunnerBase):

    class _MLPClassifier(BaseEstimator):
        def __init__(self, runner, **kwargs):
            self.runner = runner
            self.mlp = MLPClassifier(**kwargs)
            self.state_callback = self.runner._save_state
            self.user_info = [(k, v) for k, v in kwargs.items()]

            for k, v in kwargs.items():
                self.runner._log_current_argument(k, v)
            # need to intercept the classifier so we can track statistics.
            if runner.generate_curves:
                if hasattr(self.mlp, '_update_no_improvement_count'):
                    self._mlp_update_no_improvement_count = self.mlp._update_no_improvement_count
                    self.mlp._update_no_improvement_count = self._update_no_improvement_count_intercept
                if hasattr(self.mlp, '_loss_grad_lbfgs'):
                    self._mlp_loss_grad_lbfgs = self.mlp._loss_grad_lbfgs
                    self.mlp._loss_grad_lbfgs = self._loss_grad_lbfgs_intercept

        def __getattr__(self, item):
            return self.mlp.__getattribute__(item)

        def get_params(self, deep=True):
            out = super().get_params()
            out.update(self.mlp.get_params())
            return out

        def fit(self, x_train, y_train=None):
            self.runner._start_run_timing()
            # make initial callback
            self._invoke_runner_callback()
            return self.mlp.fit(x_train, y_train)

        def predict(self, x_test):
            return self.mlp.predict(x_test)

        def _update_no_improvement_count_intercept(self, early_stopping, x_val, y_val):
            self._invoke_runner_callback()
            return self._mlp_update_no_improvement_count(early_stopping, x_val, y_val)

        def _loss_grad_lbfgs_intercept(self, packed_coef_inter, x, y, activations, deltas, coef_grads, intercept_grads):
            self._invoke_runner_callback()
            return self._mlp_loss_grad_lbfgs(packed_coef_inter, x, y, activations, deltas,
                                             coef_grads, intercept_grads)

        def _invoke_runner_callback(self):
            iterations = self.mlp.n_iter_ if hasattr(self.mlp, 'n_iter_') else 0
            no_improvement_count = self.mlp._no_improvement_count if hasattr(self.mlp, '_no_improvement_count') else 0

            done = (self.mlp.early_stopping and (no_improvement_count > self.mlp.n_iter_no_change) or
                    iterations == self.mlp.max_iter)

            state = self.mlp.coefs_ if hasattr(self.mlp, 'coefs_') else []
            fitness = self.mlp.loss_ if hasattr(self.mlp, 'loss_') else 0
            curve = self.mlp.loss_curve_ if hasattr(self.mlp, 'loss_curve_') else [0]
            return self.state_callback(iteration=iterations,
                                       state=state,
                                       fitness=fitness,
                                       user_data=self.user_info,
                                       attempt=no_improvement_count,
                                       done=done,
                                       curve=curve)

    def __init__(self, x_train, y_train, x_test, y_test, experiment_name, seed, iteration_list,
                 grid_search_parameters, early_stopping=False,
                 generate_curves=True, output_directory=None):
        super().__init__(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                         experiment_name=experiment_name,
                         seed=seed,
                         iteration_list=iteration_list,
                         grid_search_parameters=grid_search_parameters,
                         generate_curves=generate_curves,
                         output_directory=output_directory)

        # build the classifier
        self.classifier = self._MLPClassifier(runner=self,
                                              shuffle=True,
                                              random_state=seed,
                                              verbose=False,
                                              warm_start=False,
                                              early_stopping=early_stopping)

        self.classifier.runner = self

    @staticmethod
    def build_grid_search_parameters(grid_search_parameters, **kwargs):
        # extract nn parameters
        all_grid_search_parameters = _NNRunnerBase.build_grid_search_parameters(grid_search_parameters, **kwargs)
        # make sure activation set is the right type.
        if 'activation' in all_grid_search_parameters:
            activation_set = all_grid_search_parameters['activation']
            for i in range(len(activation_set)):
                a = activation_set[i]
                if a == act.relu:
                    activation_set[i] = 'relu'
                elif a == act.sigmoid:
                    activation_set[i] = 'logistic'
                elif a == act.tanh:
                    activation_set[i] = 'tanh'
                elif a == act.identity:
                    activation_set[i] = 'identity'
                elif a == act.softmax:
                    activation_set[i] = 'softmax'
        return all_grid_search_parameters
