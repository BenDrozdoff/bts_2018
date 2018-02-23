# coding utf-8

from sklearn.base import TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder


class AttributeExtractor(TransformerMixin):
    def __init__(self, attr_name):
        self.id_attr_name = attr_name

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return [getattr(sample, self.attr_name) for sample in x]


class GameAttributeExtractor(TransformerMixin):
    def __init__(self, game_id_map, game_attr_name):
        self.game_id_map = game_id_map
        self.game_attr_name = game_attr_name

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return [getattr(game_id_map[sample.game_id], self.game_attr_name)
                for sample in x]


class HomeAwayExtractor(TransformerMixin):
    def __init__(self, game_id_map):
        self.game_id_map = game_id_map

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        results = list()
        for sample in x:
            game = game_id_map[sample.game_id]
            if sample.team == game.home_team:
                results.append(1)
            elif sample.team == game.away_team:
                results.append(0)
            else raise Exception('Team name {} does not match home team {} or away team {}'.format(
                sample.team, game.home_team, game.away_team))
        return results


def create_player_id_pipeline():
    return Pipeline([('player_id', AttributeExtractor('player_id')),
                     ('onehot', OneHotEncoder())])


def create_feature_matrix(game_id_map):
    return FeatureUnion(
        ('player_id', create_player_id_pipeline()),
        ('is_home', Pipeline([('is_home', HomeAwayExtractor(game_id_map))])),
        ('stadium', Pipeline([
            ('home_team_name', GameAttributeExtractor(game_id_map, 'home_team')),
            ('onehot', OneHotEncoder())]))
    )
