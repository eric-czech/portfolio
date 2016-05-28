import numpy as np
import pandas as pd
from IPython.core.debugger import Tracer


def add_benchmarks(d, groups):
    def add_benchmark(data, group_cols, numeric_cols, name):
        res = data.copy()

        # Remove records with null values for any of the group keys
        mask = res[group_cols].apply(lambda x: np.all(x.notnull()), axis=1).values
        res = res[mask]

        def benchmark(g):
            def clean(v):
                return v.apply(lambda x: x if np.isfinite(x) and not pd.isnull(x) else np.nan)
            return g.apply(clean).mean()
        bench = res.groupby(group_cols)[numeric_cols].apply(benchmark)
        bench.columns = ['Bench{}.{}'.format(name, c) for c in bench]
        bench = bench.reset_index()

        res = pd.merge(res, bench, on=group_cols, how='left')
        return res

    idx_cols = list(d.index.names)
    val_cols = list(d.columns)

    d_bench = d.copy().reset_index()

    for group_name in groups:
        d_bench = add_benchmark(d_bench, groups[group_name], val_cols, group_name)
        # d_bench = add_benchmark(d_bench, ['Date', 'Country'], val_cols, 'Country')

    return d_bench.set_index(idx_cols)


def add_calculated_quantities(d):
    c_expense = [c for c in d if c.startswith('Expenses')]
    if c_expense:
        d['ExpensesAll'] = d[c_expense].sum(axis=1)

    c_income = [c for c in d if c.startswith('Income')]
    if c_income:
        d['IncomeAll'] = d[c_income].sum(axis=1)
        d['IncomeNet'] = d['IncomeAll'] - d['ExpensesAll']

    c_cash = [c for c in d if c.startswith('Cash')]
    if c_cash:
        c_cash_begin = ['CashInBankBeginning', 'CashOnHandBeginning']
        d['CashBeginning'] = d[c_cash_begin].sum(axis=1)
        c_cash_end = ['CashInBankEnd', 'CashOnHandEnd']
        d['CashEnd'] = d[c_cash_end].sum(axis=1)
        d['CashNet'] = d['CashEnd'] - d['CashBeginning']

    c_water = [c for c in d if c.startswith('QuantityWater')]
    if c_water:
        d['QuantityWaterAll'] = d[c_water].sum(axis=1)

    return d


def log_transform_water_quality_measurements(d):
    wq_all = [c for c in d if c.startswith('WQ_')]
    wq_no_log = [
        'WQ_Alkalinity', 'WQ_Fecal Coliforms', 'WQ_Hardness',
        'WQ_Temperature', 'WQ_Total Coliforms'
    ]
    wq_log = np.setdiff1d(wq_all, wq_no_log)
    d[wq_log] = d[wq_log].applymap(lambda x: None if pd.isnull(x) else np.log10(x))
    d = d.rename(columns=lambda c: '{}_Log'.format(c) if c in wq_log else c)
    return d
