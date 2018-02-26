# coding UTF-8

from datetime import datetime
from data_extraction import load_hitter_games_by_date_range
from feature_extraction import create_feature_matrix, create_response_pipeline
from keras.models import Sequential
from keras.layers import Activation, Dense

game_id_map, hitter_games = load_hitter_games_by_date_range(
    datetime(year=2017, month=4, day=2),
    datetime(year=2017, month=10, day=2), min_pa = 3)

feature_pipeline = create_feature_matrix(game_id_map)
response_pipeline = create_response_pipeline()
feature_matrix = feature_pipeline.fit_transform(hitter_games)
response_vector = response_pipeline.fit_transform(hitter_games)

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=feature_matrix.shape[1]))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
model.fit(feature_matrix, response_vector, epochs=10, batch_size=32, validation_split=0.2)