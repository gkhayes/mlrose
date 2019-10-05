import sklearn.metrics as skmt
import sklearn.model_selection as skms


class GridSearchMixin:

    @staticmethod
    def _perform_grid_search(classifier, x_train, y_train, cv, parameters, n_jobs=1):
        scorer = GridSearchMixin.make_scorer()
        search_results = skms.GridSearchCV(classifier,
                                           parameters,
                                           cv=cv,
                                           scoring=scorer,
                                           n_jobs=n_jobs,
                                           return_train_score=True,
                                           verbose=False)
        search_results.fit(x_train, y_train)
        return search_results

    @staticmethod
    def make_scorer():
        scorer = skmt.make_scorer(skmt.balanced_accuracy_score)
        return scorer
