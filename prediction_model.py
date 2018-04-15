# coding UTF-8

import dill
import mlbgame
import pandas as pd

from datetime import datetime
from data_extraction import load_hitter_games_by_date_range
from feature_extraction import create_feature_matrix, create_response_pipeline
from models import Game

from keras.models import Sequential
from keras.layers import Activation, Dense
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split

def build_and_fit_model(
        train_start=datetime(year=2017, month=4, day=2),
        train_end=datetime(year=2017, month=10, day=2), 
        train_min_pa=3,
        train_test_split=0.2,
        random_state=37,
        epochs=10,
        batch_size=32):

        predictor = BeatTheStreakPredictor(train_start, train_end, train_min_pa, train_test_split,
        random_state, epochs, batch_size)

        predictor.gather_data()
        predictor.fit_transform(predictor.hitter_games)

        predictor.create_and_fit_prediction_model()

        return predictor


def deserialize(version_number=0):
    file_name = 'beat_the_streak_predictor_v' + str(version_number) + '.pkl'
    file = open(file_name, 'r')
    predictor = dill.load(file)
    file.close()
    return predictor


class BeatTheStreakPredictor(TransformerMixin):
    def __init__(
        self, 
        train_start=datetime(year=2017, month=4, day=2),
        train_end=datetime(year=2017, month=10, day=2), 
        train_min_pa=3,
        train_test_split=0.2,
        random_state=37,
        epochs=10,
        batch_size=32):

        self.train_start = train_start
        self.train_end = train_end
        self.train_min_pa = train_min_pa
        self.train_test_split = train_test_split
        self.random_state = random_state
        self.epochs = epochs
        self.batch_size = batch_size

    def create_and_fit_prediction_model(self):
        self.model = Sequential()
        self.model.add(Dense(32, activation='relu', input_dim=self.feature_matrix.shape[1]))
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
        self.model.fit(
            self.X_train, self.y_train, self.epochs, self.batch_size, validation_data=(self.X_test, self.y_test))
    
    def fit(self, x, y=None):
        self.feature_pipeline = create_feature_matrix(self.game_id_map)
        self.response_pipeline = create_response_pipeline()
        self.feature_pipeline.fit(x)
        self.response_pipeline.fit(x)
        return self
    
    def gather_data(self):
        self.game_id_map, self.hitter_games = load_hitter_games_by_date_range(
            self.train_start, self.train_end, self.train_min_pa
        )

    def predict_todays_games(self, n_predict=10):
        # I kind of hate this, because I'm predicting for the entire 40 man roster of a given team.
        # However, this allows for a non-time sensitive prediction for a given day.
        # Optimally this would include a lineup scraper, but this was the next best.

        today = datetime.today()
        todays_games = [Game.from_mlbgame(mlb_game) for mlb_game in 
            mlbgame.day(year=today.year, month=today.month, day=today.day)]

        today_players = []

        for game in todays_games:
            self.game_id_map[game.game_id] = game
            today_players.extend(game.retrieve_active_players())

        test_matrix = self.feature_pipeline.transform(today_players)

        today_predictions = pd.Series(
            self.model.predict_proba(test_matrix).flatten(), 
            index=[player.player_id for player in today_players])

        return today_predictions.sort_values(ascending=False)[:n_predict]

    def serialize(self, version_number=0):
        file_name = 'beat_the_streak_predictor_v' + str(version_number) + '.pkl'
        file = open(file_name, 'wb')
        dill.dump(self, file)
        file.close()

    def transform(self, x):
        self.feature_matrix = self.feature_pipeline.transform(x)
        self.response_vector = self.response_pipeline.transform(x)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.feature_matrix, self.response_vector, test_size=0.2, random_state=37
        ) 
        