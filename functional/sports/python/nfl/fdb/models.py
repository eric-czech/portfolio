__author__ = 'eczech'


from sklearn.base import BaseEstimator, clone


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
X_res['PlayerId'] = encoder.fit_transform(X_res.reset_index()['Attr:Player:PlayerLink'])

class PlayerRegressor(BaseEstimator):


    def __init__(self, overall_regressor, player_regressor):
        self.overall_regressor = overall_regressor
        self.player_regressor = player_regressor
        self.models = {}
        self.weights = {}

    def get_params(self, deep=True):
        return {
            'player_regressor': self.player_regressor,
            'overall_regressor': self.overall_regressor
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

    def fit(self, X, y):
        X = X.copy()
        self.models = {}
        self.weights = {}
        self.models['__overall__'] = clone(self.overall_regressor).fit(X.drop('PlayerId', axis=1), y)
        X['__response__'] = y.values
        for player, grp in X.groupby('PlayerId'):
            clf = clone(self.player_regressor)
            y_player = grp['__response__'].copy()
            X_player = grp.drop(['PlayerId', '__response__'], axis=1)
            self.models[str(int(player))] = clf.fit(X_player, y_player)
            self.weights[str(int(player))] = len(grp)
        return self

    def predict(self, X):
        y = []
        self.predict_ct = {'player_model': 0, 'overall_model': 0}
        for i, r in X.iterrows():
            player = str(int(r['PlayerId']))
            X_row = r.drop(['PlayerId'])

            v = self.models['__overall__'].predict(X_row)[0]
            self.predict_ct['overall_model'] += 1

            if player in self.models:
                self.predict_ct['player_model'] += 1
                w = self.weights[player]
                vp = self.models[player].predict(X_row)[0]
                w = .75 if w > 25 else 0
                v = (1-w) * v + w * vp
            y.append(v)
        return np.array(y)


from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score

o_clf = ExtraTreesRegressor(n_estimators=250, max_depth=5, min_samples_leaf=9)
p_clf = GradientBoostingRegressor()
clf = PlayerRegressor(o_clf, p_clf)
cv = StratifiedKFold(X_res['PlayerId'], 8)
scores = cross_val_score(clf, X_res, y_res, cv=cv, scoring='mean_squared_error')
#clf.fit(X_res.head(100).copy(), y_res.head(100).copy())
#clf.predict(X_res.head(100).copy())

sc = np.sqrt(np.abs(scores))
sc, np.median(sc), np.max(sc), np.min(sc)





from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
X_res['TeamId'] = encoder.fit_transform(X_res.reset_index()['Attr:Player:TeamId'])

class TeamRegressor(BaseEstimator):


    def __init__(self, overall_regressor, team_regressor):
        self.overall_regressor = overall_regressor
        self.team_regressor = team_regressor
        self.models = {}
        self.weights = {}

    def get_params(self, deep=True):
        return {
            'team_regressor': self.team_regressor,
            'overall_regressor': self.overall_regressor
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

    def fit(self, X, y):
        X = X.copy()
        self.models = {}
        self.weights = {}
        self.models['__overall__'] = clone(self.overall_regressor).fit(X.drop('TeamId', axis=1), y)
        X['__response__'] = y.values
        for player, grp in X.groupby('TeamId'):
            clf = clone(self.team_regressor)
            y_player = grp['__response__'].copy()
            X_player = grp.drop(['TeamId', '__response__'], axis=1)
            self.models[str(int(player))] = clf.fit(X_player, y_player)
            self.weights[str(int(player))] = len(grp)
        return self

    def predict(self, X):
        y = []
        self.predict_ct = {'team_model': 0, 'overall_model': 0}
        for i, r in X.iterrows():
            team = str(int(r['TeamId']))
            X_row = r.drop(['TeamId'])

            v = self.models['__overall__'].predict(X_row)[0]
            self.predict_ct['overall_model'] += 1

            if team in self.models:
                self.predict_ct['team_model'] += 1
                w = self.weights[team]
                vp = self.models[team].predict(X_row)[0]
                w = .5# if w > 25 else 0
                v = (1-w) * v + w * vp
            y.append(v)
        return np.array(y)


from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score

o_clf = ExtraTreesRegressor(n_estimators=250, max_depth=5, min_samples_leaf=9)
t_clf = GradientBoostingRegressor()
clf = TeamRegressor(o_clf, t_clf)
cv = StratifiedKFold(X_res['TeamId'], 8)
scores = cross_val_score(clf, X_res, y_res, cv=cv, scoring='mean_squared_error')