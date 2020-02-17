import sklearn.metrics as skmt
import sklearn.model_selection as skms
import inspect


class GridSearchMixin:
    def __init__(self, scorer_method=None):
        self._scorer_method = skmt.balanced_accuracy_score if scorer_method is None else scorer_method
        self._params = inspect.signature(self._scorer_method)
        self._get_y_argmax = False

    def _perform_grid_search(self, classifier, x_train, y_train, cv, parameters, n_jobs=1, verbose=False):
        scorer = self.make_scorer()
        search_results = skms.GridSearchCV(classifier,
                                           parameters,
                                           cv=cv,
                                           scoring=scorer,
                                           n_jobs=n_jobs,
                                           return_train_score=True,
                                           verbose=verbose)
        search_results.fit(x_train, y_train)
        return search_results

    def make_scorer(self):
        scorer = skmt.make_scorer(self._grid_search_score_intercept)
        return scorer

    def score(self, **kwargs):
        score = self._grid_search_score_intercept(**kwargs)
        return score

    def _grid_search_score_intercept(self, y_pred, y_true, **kwargs):
        cleaned_kwargs = {k: v for k, v in kwargs.items() if k in list(inspect.signature(self._scorer_method).parameters)}

        # workaround for odd usage from sklearn
        if not hasattr(self, '_get_y_argmax'):
            self._get_y_argmax = False

        if not self._get_y_argmax and len(y_pred.shape) > 1 and len(y_true.shape) > 1:
            try:
                return self._scorer_method(y_pred=y_pred, y_true=y_true, **cleaned_kwargs)
            except:
                self._get_y_argmax = True

        if self._get_y_argmax:
            y_pred = y_pred.argmax(axis=1)
            y_true = y_true.argmax(axis=1)

        return self._scorer_method(y_pred=y_pred, y_true=y_true, **cleaned_kwargs)
