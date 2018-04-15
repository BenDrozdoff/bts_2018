# coding utf-8

import numpy as np

from sklearn.base import TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder


def hitter_home_or_away(hitter_game, game, home_output, away_output):
    if hitter_game.team == game.home_team:
        return home_output
    elif hitter_game.team == game.away_team:
        return away_output
    else:
        raise Exception('Team name {} does not match home team {} or away team {}'.format(
            hitter_game.team, game.home_team, game.away_team))


class LabelledPipeline(Pipeline):

    def get_feature_names(self):
        last_transformer = self.steps[-1]
        if hasattr(last_transformer[1], 'get_feature_names'):
            return last_transformer[1].get_feature_names()
        else:
            raise AttributeError('Transformer {} of class {} does not provide method get_feature_names'.format(
                last_transformer[0], last_transformer[1]))


class OneHotEncoderWithFeatureNames(OneHotEncoder):
    def __init__(self, label_prepend, handle_unknown='ignore'):
        super().__init__()
        self.label_prepend = label_prepend
        self.handle_unknown = handle_unknown

    def get_feature_names(self):
        return ['{}: {}'.format(self.label_prepend, feature) for feature in list(self.active_features_)]



class AttributeExtractor(TransformerMixin):
    def __init__(self, attr_name):
        self.attr_name = attr_name

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return np.array([getattr(sample, self.attr_name) for sample in x]).reshape(-1, 1)


class GameAttributeExtractor(TransformerMixin):
    def __init__(self, game_id_map, game_attr_name):
        self.game_id_map = game_id_map
        self.game_attr_name = game_attr_name

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return np.array([getattr(self.game_id_map[sample.game_id], self.game_attr_name)
                         for sample in x]).reshape(-1, 1)


class TextDictEncoder(TransformerMixin):
    '''Hacky alternative to using LabelEncoder or LabelBinarizer within a pipeline
    Transforming free text into dicts and using dict vectorizer'''

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return [{string[0]: True} for string in x]


class HomeAwayExtractor(TransformerMixin):
    def __init__(self, game_id_map):
        self.game_id_map = game_id_map

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return np.array(
            [hitter_home_or_away(sample, self.game_id_map[sample.game_id], 1, 0) for sample in x]).reshape(-1, 1)

    def get_feature_names(self):
        return ['home']


class GotHitExtractor(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return np.array([1 if sample.h > 0 else 0 for sample in x])


class OpponentExtractor(TransformerMixin):
    def __init__(self, game_id_map, home_attr, away_attr, dtype=np.float):
        self.game_id_map = game_id_map
        self.home_attr = home_attr
        self.away_attr = away_attr
        self.dtype=dtype

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        results = np.empty(len(x), dtype=self.dtype)
        for row_index, hitter_game in enumerate(x):
            game = self.game_id_map[hitter_game.game_id]
            results[row_index] = hitter_home_or_away(
                hitter_game=hitter_game,
                game=game,
                home_output=getattr(game, self.home_attr, 0),
                away_output=getattr(game, self.away_attr, 0))
        return results.reshape(-1, 1)


def create_player_id_pipeline():
    return LabelledPipeline([('player_id', AttributeExtractor('player_id')),
                     ('onehot', OneHotEncoderWithFeatureNames('player_id'))])


def create_feature_matrix(game_id_map):
    return FeatureUnion([
        ('player_id', create_player_id_pipeline()),
        ('is_home', LabelledPipeline([('is_home', HomeAwayExtractor(game_id_map))])),
        ('stadium', LabelledPipeline([
            ('home_team_name', GameAttributeExtractor(game_id_map, 'home_team')),
            ('dictionary', TextDictEncoder()),
            ('binarize', DictVectorizer())])),
        ('opposing_starter', LabelledPipeline([
            ('opposing_starter_id', OpponentExtractor(
                game_id_map, home_attr='away_starter', away_attr='home_starter')),
            ('onehot', OneHotEncoderWithFeatureNames('opposing_starter'))])),
        ('opposing_team',  LabelledPipeline([
            ('opposing_team_name', OpponentExtractor(
                game_id_map, home_attr='away_team', away_attr='home_team', dtype=np.str)),
            ('dictionary', TextDictEncoder()),
            ('binarize', DictVectorizer())]))]
    )


def create_response_pipeline():
    return Pipeline([('got_hit', GotHitExtractor())])
