import pandas as pd
from ml.api.results.predictions import TASK_PROPERTY


def sample_tasks(d_score, limit=10, random_state=None):
    tasks = pd.Series(d_score.columns.get_level_values(TASK_PROPERTY)).drop_duplicates()
    return tasks.sample(n=min(limit, len(tasks)), random_state=random_state)
