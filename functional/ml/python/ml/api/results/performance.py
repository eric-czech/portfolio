
import pandas as pd
from ml.api.results.predictions import PREDICTION_PANEL, META_PANEL
from ml.api.results.predictions import MODEL_PROPERTY, FOLD_PROPERTY
from ml.api.results.predictions import VALUE_PRED_PREFIX, CLASS_PRED_PREFIX, CLASS_PROB_PREFIX, \
    VALUE_TRUE_PREFIX, CLASS_TRUE_PREFIX
from ml.api.results.predictions import extract
from ml.api.results import properties
from ml.api.constants import MODE_CLASSIFIER, MODE_REGRESSOR

METRIC_PROPERTY = 'Metric'
TASK_PROPERTY = 'Task'


def compute_scores(train_res, score_fn, predictions=None):

    if predictions is None:
        predictions = extract(train_res)

    d = predictions[[PREDICTION_PANEL, META_PANEL]]

    tasks = properties.get_prediction_tasks(train_res)

    def get_scores(g):
        g = g[PREDICTION_PANEL]
        res = {}
        for task in tasks:
            if train_res.mode == MODE_CLASSIFIER:
                y_class = g[CLASS_PRED_PREFIX + ':' + task]
                y_true = g[CLASS_TRUE_PREFIX + ':' + task]
                k = CLASS_PROB_PREFIX + ':' + task
                y_proba = g.filter(regex=k)
                y_proba = y_proba if len(y_proba.columns) > 0 else None
                scores = score_fn(y_true, y_class, y_proba)

            elif train_res.mode == MODE_REGRESSOR:
                y_pred = g[VALUE_PRED_PREFIX + ':' + task]
                y_true = g[VALUE_TRUE_PREFIX + ':' + task]
                scores = score_fn(y_true, y_pred)
            else:
                raise NotImplementedError('Training mode "{}" not implemented'.format(train_res.mode))
            res.update({(k, task): scores[k] for k in scores})
        res = pd.Series(res)
        res.index = pd.MultiIndex.from_tuples(res.index.values)
        return res
    # Group by model name and fold id, and compute scores for each task
    d = d.groupby([(META_PANEL, MODEL_PROPERTY), (META_PANEL, FOLD_PROPERTY)]).apply(get_scores)

    # Cleanup margin names
    d.index.names = [n[1] for n in d.index.names]
    d.columns.names = [METRIC_PROPERTY, TASK_PROPERTY]

    return d


def sample_tasks(scores, limit=10, random_state=None):
    return pd.Series(scores.columns.get_level_values(TASK_PROPERTY))\
        .drop_duplicates().sample(n=limit, random_state=random_state)


def plot_scores(scores, metrics=None, tasks=None, figsize=2, figaspect=5, legend_margin=(1.13, 1)):
    """
    Generates boxplot of performance metrics

    :param scores: DataFrame from `performance.compute_scores`
    :param metrics:
    :param tasks:
    :param figsize:
    :param figaspect:
    :param legend_margin:
    :return:
    """
    import seaborn as sns

    # Stack model scores so that instead of one column per task + metric, we have a single row
    # for each model, fold, task, and metric
    d_score = scores.stack(level=[TASK_PROPERTY, METRIC_PROPERTY])
    d_score.name = 'Value'
    d_score = d_score.reset_index().reset_index(drop=True)

    # Filter to only the given metrics, if any were provided
    if metrics is not None:
        d_score = d_score[d_score[METRIC_PROPERTY].isin(metrics)]

    # Filter to only the given tasks, if any were provided
    if tasks is not None:
        d_score = d_score[d_score[TASK_PROPERTY].isin(tasks)]

    # Create a boxplot per task (in rows) with model names on the x axis and colors denoting metrics
    g = sns.FacetGrid(d_score, row=TASK_PROPERTY, size=figsize, aspect=figaspect, legend_out=False)

    # Map grid to boxplot function
    def plot_data(*args, **kwargs):
        sns.boxplot(data=kwargs['data'], x=MODEL_PROPERTY, y='Value', hue=METRIC_PROPERTY)
    g.map_dataframe(plot_data)
    [ax.legend(bbox_to_anchor=legend_margin) for ax in g.axes.ravel()]

    # Return the plot grid and the underlying data
    return g, d_score
