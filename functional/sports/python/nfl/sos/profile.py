__author__ = 'eczech'

import numpy as np


def rm_na(x):
    return x[~x.isnull()]


def get_wavg_stat(value_field, weight_field):
    def get_value(x):
        w = rm_na(x[weight_field])
        w = np.repeat(1/len(w), len(w)) if w.sum() == 0 else w/w.sum()
        return np.dot(rm_na(x[value_field]), w)
    return get_value


def get_passing_yds(x):
    v1 = rm_na(x['passing:Yds'])
    v2 = rm_na(x['passing:Loss'])
    return np.sum(v1 - v2)


def get_total_yds(x):
    return get_passing_yds(x) + rm_na(x['rushing:Yds']).sum()

PROFILE_PASSING = {
    'input_fields': [
        'passing:Yds', 'passing:YPA', 'passing:Rate', 'passing:Att', 'passing:Loss'
    ],
    'output_fields': {
        'passing:Yds': get_passing_yds,
        'passing:YPA': get_wavg_stat('passing:YPA', 'passing:Att'),
        'passing:Rate': get_wavg_stat('passing:Rate', 'passing:Att')
    },
    'weights': {
        'passing:Yds': .5,
        'passing:YPA': .1,
        'passing:Rate': .4
    },
    'name': 'passing',
    'primary_output': 'passing:Yds'
}


PROFILE_RUSHING = {
    'input_fields': [
        'rushing:Yds', 'rushing:Att', 'rushing:Avg'
    ],
    'output_fields': {
        'rushing:Yds': lambda x: rm_na(x['rushing:Yds']).sum(),
        'rushing:Avg': get_wavg_stat('rushing:Avg', 'rushing:Att'),
        'rushing:Att': lambda x: rm_na(x['rushing:Att']).sum()
    },
    'weights': {
        'rushing:Yds': .8,
        'rushing:Avg': .1,
        'rushing:Att': .1
    },
    'name': 'rushing',
    'primary_output': 'rushing:Avg'
}

PROFILE_RECEIVING = {
    'input_fields': [
        'receiving:Yds', 'receiving:Rec', 'receiving:Avg'
    ],
    'output_fields': {
        'receiving:Yds': lambda x: rm_na(x['receiving:Yds']).sum(),
        'receiving:Avg': get_wavg_stat('receiving:Avg', 'receiving:Rec'),
        'receiving:Rec': lambda x: rm_na(x['receiving:Rec']).sum()
    },
    'weights': {
        'receiving:Yds': .8,
        'receiving:Avg': .1,
        'receiving:Rec': .1
    },
    'name': 'receiving',
    'primary_output': 'receiving:Yds'
}

PROFILE_OVERALL = {
    'input_fields': [
        'passing:Yds', 'passing:Loss', 'rushing:Yds'
    ], 'output_fields': {
        'overall:Yds': get_total_yds
    }, 'weights': {
        'overall:Yds': 1,
    },
    'name': 'overall',
    'primary_output': 'overall:Yds'
}


