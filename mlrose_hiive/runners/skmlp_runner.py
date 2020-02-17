import mlrose_hiive.neural.activation as act
import sklearn.metrics as skmt

from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator
from mlrose_hiive.decorators import short_name
from mlrose_hiive.runners._nn_runner_base import _NNRunnerBase


@short_name('skmlp')
class SKMLPRunner(_NNRunnerBase):
    class _MLPClassifier(BaseEstimator):
        def __init__(self, runner, **kwargs):
            self.runner = runner
            self.mlp = MLPClassifier(**kwargs)
            self.state_callback = self.runner._save_state
            self.fit_started_ = False
            self.user_info_ = None
            self.kwargs_ = kwargs

            self.loss_ = 1.
            self.state_ = None
            self.curve_ = []

            # need to intercept the classifier so we can track statistics.
            if runner.generate_curves:
                if hasattr(self.mlp, '_update_no_improvement_count'):
                    self._mlp_update_no_improvement_count = self.mlp._update_no_improvement_count
                    self.mlp._update_no_improvement_count = self._update_no_improvement_count_intercept
                if hasattr(self.mlp, '_loss_grad_lbfgs'):
                    self._mlp_loss_grad_lbfgs = self.mlp._loss_grad_lbfgs
                    self.mlp._loss_grad_lbfgs = self._loss_grad_lbfgs_intercept

        def __getattr__(self, item, default=None):
            if 'mlp' in self.__dict__ and hasattr(self.__dict__['mlp'], item):
                return self.__dict__['mlp'].__getattr__(item, default)
            return self.__dict__[item] if item in self.__dict__ else default

        def __setattr__(self, item, value):
            if 'mlp' in self.__dict__ and hasattr(self.__dict__['mlp'], item):
                self.__dict__['mlp'].__setattr__(item, value)
            self.__dict__[item] = value

        def get_params(self, deep=True):
            out = super().get_params()
            out.update(self.mlp.get_params())
            # exclude any that end with an underscore
            out = {k: v for (k, v) in out.items() if not k[-1] == '_'}
            return out

        def fit(self, x_train, y_train=None):
            self.fit_started_ = True
            self.runner._start_run_timing()
            # make initial callback
            self._invoke_runner_callback()
            return self.mlp.fit(x_train, y_train)

        def predict(self, x_test):
            return self.mlp.predict(x_test)

        def _update_no_improvement_count_intercept(self, early_stopping, x_val, y_val):
            ret = self._mlp_update_no_improvement_count(early_stopping, x_val, y_val)
            self._state = self.mlp.coefs_ if hasattr(self.mlp, 'coefs_') else []
            self.loss_ = self.mlp.loss_ if hasattr(self.mlp, 'loss_') else 0
            if hasattr(self.mlp, 'loss_curve_'):
                self.curve_ = self.mlp.loss_curve_
            else:
                self.curve_.append(self.loss_)
            self._invoke_runner_callback()
            return ret

        def _loss_grad_lbfgs_intercept(self, packed_coef_inter, x, y, activations, deltas, coef_grads, intercept_grads):
            f, g = self._mlp_loss_grad_lbfgs(packed_coef_inter, x, y, activations, deltas,
                                             coef_grads, intercept_grads)
            self.loss_ = f
            self.state_ = g
            self.curve_.append(self.loss_)
            self._invoke_runner_callback()
            return f, g

        def _invoke_runner_callback(self):
            iterations = self.mlp.n_iter_ if hasattr(self.mlp, 'n_iter_') else 0
            no_improvement_count = self.mlp._no_improvement_count if hasattr(self.mlp, '_no_improvement_count') else 0

            done = (self.mlp.early_stopping and (no_improvement_count > self.mlp.n_iter_no_change) or
                    iterations == self.mlp.max_iter)

            # check for early abort.
            if self.runner.has_aborted():
                return self
            if self.user_info_ is None:
                self.user_info_ = [(k, self.__dict__[k]) for k in self.kwargs_.keys() if hasattr(self, k)]
                for k, v in self.user_info_:
                    self.runner._log_current_argument(k, v)

            return self.state_callback(iteration=iterations,
                                       state=self.state_,
                                       fitness=self.loss_,
                                       user_data=self.user_info_,
                                       attempt=no_improvement_count,
                                       done=done,
                                       curve=self.curve_)

    def __init__(self, x_train, y_train, x_test, y_test, experiment_name, seed, iteration_list,
                 grid_search_parameters, grid_search_scorer_method=skmt.balanced_accuracy_score,
                 early_stopping=True, max_attempts=500, n_jobs=1, cv=5,
                 generate_curves=True, output_directory=None, replay=False, **kwargs):

        # take a copy of the grid search parameters
        grid_search_parameters = {**grid_search_parameters}

        # hack for compatibility purposes
        if 'max_iters' in grid_search_parameters:
            grid_search_parameters['max_iter'] = grid_search_parameters.pop('max_iters')

        if 'max_attempts' in grid_search_parameters:
            grid_search_parameters['n_iter_no_change'] = grid_search_parameters.pop('max_attempts')

        super().__init__(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                         experiment_name=experiment_name,
                         seed=seed,
                         iteration_list=iteration_list,
                         grid_search_parameters=grid_search_parameters,
                         grid_search_scorer_method=grid_search_scorer_method,
                         generate_curves=generate_curves,
                         output_directory=output_directory,
                         replay=replay,
                         n_jobs=n_jobs,
                         cv=cv)

        # build the classifier
        self.classifier = self._MLPClassifier(runner=self,
                                              shuffle=True,
                                              random_state=seed,
                                              verbose=False,
                                              warm_start=False,
                                              early_stopping=early_stopping,
                                              n_iter_no_change=max_attempts,
                                              **kwargs)

        self.classifier.runner = self

    @staticmethod
    def build_grid_search_parameters(grid_search_parameters, **kwargs):
        # extract nn parameters
        all_grid_search_parameters = _NNRunnerBase.build_grid_search_parameters(grid_search_parameters, **kwargs)
        # make sure activation set is the right type.
        if 'activation' in all_grid_search_parameters:
            activation_set = list(all_grid_search_parameters['activation'])
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
            all_grid_search_parameters['activation'] = activation_set
        return all_grid_search_parameters
