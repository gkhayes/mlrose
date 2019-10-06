import sklearn.metrics as skmt
import sklearn.model_selection as skms


class GridSearchMixin:
    _scorer_method = skmt.balanced_accuracy_score
    @staticmethod
    def _perform_grid_search(classifier, x_train, y_train, cv, parameters, n_jobs=1, verbose=False):
        scorer = GridSearchMixin.make_scorer()
        search_results = skms.GridSearchCV(classifier,
                                           parameters,
                                           cv=cv,
                                           scoring=scorer,
                                           n_jobs=n_jobs,
                                           return_train_score=True,
                                           verbose=verbose)
        search_results.fit(x_train, y_train)
        return search_results

    @staticmethod
    def make_scorer():
        scorer = skmt.make_scorer(GridSearchMixin._scorer_method)
        return scorer

    @staticmethod
    def score(y_true, y_pred, sample_weight=None, adjusted=False):
        return GridSearchMixin._scorer_method(y_true, y_pred,
                                              sample_weight=sample_weight,
                                              adjusted=adjusted)
