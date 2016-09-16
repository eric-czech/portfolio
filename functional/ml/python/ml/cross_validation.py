
import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedKFold, KFold, PredefinedSplit


def get_entity_cv(d, id_col, id_label_func=None, validate_assignments=True, **cv_kwargs):
    """
    Returns a CV strategy based on entity-based stratified, leave P entity out (exclusively) sampling

    More specifically, this strategy works like the following:
        1. Assume the data given has an "entity id" where each entity has multiple records
        2. Optionally, determine a single outcome label for each entity (if id_label_func not None)
        3. Creates an entity-based CV split, which will be stratified if entities were given outcomes
        4. Returns a predefined split on the entire dataset using the entity-based split

    :param d: Modeling DataFrame with entity id in "id_col" column
    :param id_col: Name of entity identifier column
    :param id_label_func: Optional function accepting a DataFrame with all data for one entity and returning
        a single, scalar label that should be used to stratify that entity.
        Example: lambda g: g[response].value_counts().argmax() # Return label as most common response value
    :param validate_assignments: Indicates whether or not to gaurantee that the assignment of entities to
        folds is unique (there's no reason to disable this except for repatitive executions on large datasets)
    :param cv_kwargs: Arguments for entity CV split function (i.e. n_folds, shuffle, random_state, etc for
        either KFold or StratifiedKFold)
    """
    assert np.all(d[id_col].notnull()), 'Entity ID value cannot be null'

    # Determine the label for each entity, if an outcome labelling function was given,
    # and either create an arbitrary n_fold split or a stratified one (if outcome labeller given)
    if id_label_func is None:
        y = d.groupby(id_col).apply(id_label_func)
        assert np.all(y.notnull()), 'Entity outcome label cannot be null'
        entity_cv = StratifiedKFold(y.values, **cv_kwargs)
    else:
        y = d.groupby(id_col).size()
        entity_cv = KFold(len(y), **cv_kwargs)

    # Aggregate test index arrays with mapping to the fold number
    cv = []
    for i, (_, test) in enumerate(entity_cv):
        cv.append(pd.Series(np.repeat(i, len(test)), index=test))
    cv = pd.concat(cv)

    # Create a mapping of entity id to test fold number
    cv = pd.Series(cv.values, index=y.index.values[cv.index.values])

    # Ensure that no records were lost on entity splits
    assert len(cv) == len(y)
    assert len(np.unique(cv.index.values)) == len(y)
    assert np.all(pd.notnull(cv.index.values))

    # Map test fold assignments to entire dataset
    cv = d[id_col].map(cv.to_dict())

    # Ensure that no records were lost during the assignment over the whole dataset
    assert len(cv) == len(d)
    assert np.all(cv.notnull())

    # Convert fold assignments to predefined split
    cv = PredefinedSplit(cv)

    # And finally, verify that the mapping of entity ids to fold assignments is truly
    # unique and comprehensive, if configured to do so
    if validate_assignments:
        id_col_fold = id_col+'__fold'
        id_ct = d[[id_col]].assign(**{id_col_fold: cv.test_fold}).groupby(id_col)[id_col_fold].nunique()
        id_ct_freq = id_ct.value_counts()
        assert len(id_ct_freq) == 1 and id_ct_freq.index.values[0] == 1, \
            'Found at least one entity with multiple or no fold assignments.  Examples:\n{}'\
            .format(id_ct[id_ct != 1].head())

    return cv