
import pandas as pd


def has_training_data(train_res):
    return train_res.trainer_config.keep_training_data


def has_test_feature_data(train_res):
    return train_res.trainer_config.keep_test_data


def extract_resample_model_list(train_res):
    """
    Return list of all modeling results generated via resampling

    This is similar to `extract_refit_model_map` except that it returns model results from cross-validation/resampling,
    and not from refitting on the entire training set (note these are also returned as a list because there are
    multiple instances of the same model (per resample) and this is generally easier to work with as a list).

    :param train_res: Training Result
    :return: List of TrainingModelResult instances
    """
    res = []
    for resample_res in train_res.resample_results:
        for model_res in resample_res.model_results:
            res.append(model_res)
    return res


def extract_refit_model_map(train_res):
    """
    Return a dictionary of refit models on training data

    This is similar to `extract_resample_model_list` except that it is for refit models (on entire
    training set) and is returned as a dict because there are not multiple instances of the same models, per-resample

    :param train_res: Training Result
    :return: Dict with key as string name of model and value of TrainingModelResult
    """
    m = {}
    for model_res in train_res.refit_result.model_results:
        m[model_res.clf_name] = model_res
    return m


def extract_property_by_consensus(train_res, prop_fn, prop_name):
    d = []
    last = None
    all_equal = True
    for model_result in extract_resample_model_list(train_res):
        value = prop_fn(model_result)
        if last is not None and all_equal:
            all_equal = value == last
        last = value
        d.append((model_result.clf_name, model_result.fold, value))
    d = pd.DataFrame(d, columns=['clf', 'fold', 'value'])
    if not all_equal:
        raise ValueError(
            'At least one value for the property "{}" did not agree across all model results.  '
            'Values returned =\n{}'.format(prop_name, d)
        )
    return d['value'].iloc[0]


def has_single_task(train_res):
    return extract_property_by_consensus(train_res, lambda mr: len(mr.Y_names) == 1, 'has_single_task')


def get_prediction_tasks(train_res):
    return extract_property_by_consensus(train_res, lambda mr: mr.Y_names, 'prediction_tasks')
