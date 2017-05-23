
import pandas as pd
import plotly as plty
from plotly import graph_objs as go

CRIME_CATEGORIES = [
    'Alarm', 'Robbery', 'Verbal Dispute', 'Vandalism',
    'Drinking', 'Assault', 'Drugs', 'Domestic Violence',
    'Trespassing', 'Shooting', 'Suicide',
    'Missing Person', 'Gun Possession',
    'Social Services', 'Death', 'Stabbing', 'Homicide'
]


def plot_zip_stats(d, zipcode, categories=None, smooth_window=None, plot=True, title=None):
    dt = d[d['zip'] == zipcode]

    if categories is not None:
        dt = dt[dt['nature_category'].isin(categories)]

    dt = dt.set_index('calltime')

    dt = dt.groupby([pd.TimeGrouper('1MS'), 'nature_category']).size().unstack().sort_index()
    traces = []
    def smooth(v):
        if smooth_window is None:
            return v
        return v.rolling(window=smooth_window, min_periods=1, center=True).mean()

    for c in dt:
        traces.append(go.Scatter(
            x = dt.index,
            y = smooth(dt[c]),
            name=c
        ))

    dt_mean = dt.mean(axis=1)
    traces.append(go.Scatter(
        x = dt_mean.index,
        y = smooth(dt_mean),
        name = 'Average',
        mode='lines+markers'
    ))
    if title is None:
        title = 'Crime Rates in Zip {}'.format(int(zipcode))
    layout = go.Layout(title=title, yaxis=dict(title='911 Call Rate per Month'))
    fig = go.Figure(data=traces, layout=layout)
    if plot:
        plty.offline.iplt(fig)
    else:
        return fig