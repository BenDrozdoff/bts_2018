# coding utf-8

import mlbgame
import os
import pandas as pd
import time

from datetime import datetime
from models import Game, HiterGame
from selenium import webdriver;
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile;


def load_hitter_games_by_date_range(start, end):
    '''Takes in 2 Datetime values and outputs a list of HitterGame objects for the relevant dates'''
    if start > end:
        raise Exception('Start date greater than end date, please edit')
    hitter_games_out = []
    game_by_game_id = dict()
    date_range = pd.date_range(start, end)
    for date in date_range:
        mlb_games = mlbgame.day(year=date.year, day=date.day, month=date.month)
        for mlb_game in mlb_games:
            if mlb_game.game_status != 'FINAL':
                continue
            game = Game.from_mlbgame(mlb_game)
            hitters = game.retrieve_player_statistics()
            game_by_game_id[game.game_id] = game
            for hitter in hitters:
                hitter_games_out.append(hitter)
    return game_by_game_id, hitter_games_out


def scrape_steamer_projections():
	today = datetime.today()
	download_dir = os.getcwd()
	default_filepath = os.path.join(download_dir, 'Fangraphs Leaderboard.csv')
	desired_filepath = os.path.join(
    download_dir, '{}_{}_{}_steamer.csv'.format(
        today.year, today.month, today.day))

	profile = FirefoxProfile();

	profile.set_preference("browser.helperApps.neverAsk.saveToDisk", 'text/csv')
	profile.set_preference("browser.download.manager.showWhenStarting", False)
	profile.set_preference("browser.download.dir", download_dir)
	profile.set_preference(
    "browser.download.folderList",
     2)  # download to last location set

	driver = webdriver.Firefox(firefox_profile=profile);

	driver.get(
	    "https://www.fangraphs.com/projections.aspx?pos=all&stats=bat&type=steamer")
	driver.find_element_by_link_text('Export Data').click()

	# leave browser open for 10 seconds then close
	time.sleep(10)
	driver.quit()

	if os.path.isfile(default_filepath):
		os.rename(default_filepath, desired_filepath);
		print('Renamed file %s to %s' % (default_filepath, desired_filepath));
	else:
		sys.exit('Error, unable to locate file at %s' % (default_filepath)
