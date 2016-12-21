
import pandas as pd


def has_training_data(train_res):
    return train_res.trainer_config.keep_training_data


def has_test_feature_data(train_res):
    return train_res.trainer_config.keep_test_data


def extract_model_results(train_res):
    res = []
    for resample_res in train_res.resample_results:
        for model_res in resample_res.model_results:
            res.append(model_res)
    return res


def extract_property_by_consensus(train_res, prop_fn, prop_name):
    d = []
    last = None
    all_equal = True
    for model_result in extract_model_results(train_res):
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
