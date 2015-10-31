__author__ = 'eczech'


from nfl.fdb import training


class Dataset:

    def __init__(self, position, response, primary_predictor, sos_type, derived_fields):
        self.position = position
        self.response = response
        self.primary_predictor = primary_predictor
        self.sos_type = sos_type
        self.sos_damping_filter = None
        self.derived_fields = derived_fields

    def get_training_data(self, years, damping_factor_filter=None, **kwargs):
        adj_values = {self.primary_predictor: {'sos_type': self.sos_type, 'sos_damping_filter': damping_factor_filter}}
        X, y = training.get_position_data(years, self.position, self.response, adj_values, **kwargs)
        X = training.add_binary_derived_fields(X, self.derived_fields)
        X = training.add_dvoa_derived_fields(X)
        return X, y

    def get_primary_predictor(self):
        return self.primary_predictor+':Adj:None'


DATASET_QB_YDS = Dataset('QB', 'Stat:Result:passing:Yds', 'Stat:Player:passing:Yds', 'passing:defense', [
    {
        'op': 'diff',
        'result': 'Stat:Player:passing:Yds:Diff',
        'left_field': 'Stat:Player:passing:Yds:Adj:None',
        'right_field': 'Stat:Opp:QB:passing:Yds:2nd'
    }, {
        'op': 'diff',
        'result': 'Stat:Player:passing:Att:Diff',
        'left_field': 'Stat:Player:passing:Att',
        'right_field': 'Stat:Opp:QB:passing:Att:2nd'
    }, {
        'op': 'diff',
        'result': 'Stat:Player:passing:Rate:Diff',
        'left_field': 'Stat:Player:passing:Rate',
        'right_field': 'Stat:Opp:QB:passing:Rate:2nd'
    }, {
        'op': 'diff',
        'result': 'Stat:Player:passing:TD:Diff',
        'left_field': 'Stat:Player:passing:TD',
        'right_field': 'Stat:Opp:QB:passing:TD:2nd'
    }, {
        'op': 'diff',
        'result': 'Stat:Player:passing:Int:Diff',
        'left_field': 'Stat:Player:passing:Int',
        'right_field': 'Stat:Opp:QB:passing:Int:2nd'
    }, {
        'op': 'diff',
        'result': 'Stat:Player:passing:YPA:Diff',
        'left_field': 'Stat:Player:passing:YPA',
        'right_field': 'Stat:Opp:QB:passing:YPA:2nd'
    }, {
        'op': 'diff',
        'result': 'Stat:Player:rushing:Yds:Diff',
        'left_field': 'Stat:Player:rushing:Yds',
        'right_field': 'Stat:Opp:QB:rushing:Yds:2nd'
    }, {
        'op': 'diff',
        'result': 'Stat:Team:QB:passing:Yds:Diff',
        'left_field': 'Stat:Team:QB:passing:Yds:1st',
        'right_field': 'Stat:Opp:QB:passing:Yds:2nd'
    }, {
        'op': 'diff',
        'result': 'Stat:Team:RB:rushing:Yds:Diff',
        'left_field': 'Stat:Team:RB:rushing:Yds:1st',
        'right_field': 'Stat:Opp:RB:rushing:Yds:2nd'
    }
])

DATASET_RB_YDS = Dataset('RB', 'Stat:Result:rushing:Yds', 'Stat:Player:rushing:Yds', 'rushing:defense', [
    {
        'op': 'diff',
        'result': 'Stat:Player:rushing:Yds:Diff',
        'left_field': 'Stat:Player:rushing:Yds:Adj:None',
        'right_field': 'Stat:Opp:RB:rushing:Yds:2nd'
    }, {
        'op': 'diff',
        'result': 'Stat:Team:QB:rushing:Yds:Diff',
        'left_field': 'Stat:Team:QB:rushing:Yds:1st',
        'right_field': 'Stat:Opp:QB:rushing:Yds:2nd'
    }, {
        'op': 'diff',
        'result': 'Stat:Team:RB:rushing:Yds:Diff',
        'left_field': 'Stat:Team:RB:rushing:Yds:1st',
        'right_field': 'Stat:Opp:RB:rushing:Yds:2nd'
    }
])