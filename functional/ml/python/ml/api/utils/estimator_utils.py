

from sklearn.model_selection._search import BaseSearchCV
from sklearn.pipeline import Pipeline


def resolve_estimator(clf):
    """
    Returns the underlying learing model for an estimator instance

    For example, an estimator may be a Pipeline or GridSearchCV instance
    and for either case this method would return the learning estimator in
    the Pipeline (if named 'est' or 'clf') or the "best estimator" within
    the GridSearchCV instance.

    Note that the estimator will be returned as is if it is an estimator wrapper
    that is not yet supported.

    :param clf: Estimator to find learning estimator within
    :return: Learning estimator instance
    """

    # If model is grid search, fetch underlying, best estimator
    if isinstance(clf, BaseSearchCV):
        return clf.best_estimator_

    # If model is pipeline, resolved contained estimator
    pipe_clf = resolve_estimator_from_pipeline(clf)
    if pipe_clf is not None:
        return resolve_estimator(pipe_clf)

    # Otherwise, return as is
    return clf


def resolve_estimator_from_pipeline(clf):
    """
    Returns the estimator within a pipeline named as 'est' or 'clf'

    :param clf: The estimator instance (possibly a pipeline) to resolve
    :return: The estimator found or None if the given estimator was not a pipeline
        or a step named 'est' or 'clf' as not found.
    """
    if isinstance(clf, Pipeline):
        steps = clf.named_steps
        for name in ['clf', 'est']:
            if name in steps:
                return clf.named_steps[name]
    return None
