# coding utf-8

import mlbgame


class Game():
    def __init__(self, game_id, home_team, away_team, date):
        self.game_id = game_id
        self.home_team = home_team
        self.away_team = away_team
        self.date = date

    def __repr__(self):
        return self.game_id

    def from_mlbgame(mlb_game_object):
        return Game(
            game_id=mlb_game_object.game_id,
            home_team=mlb_game_object.home_team,
            away_team=mlb_game_object.away_team,
            date=mlb_game_object.date)

    def retrieve_player_statistics(self):
        player_statistics = mlbgame.player_stats(self.game_id)
        self.home_starter = player_statistics.home_pitching[0].id
        self.away_starter = player_statistics.away_pitching[0].id
        away_batters = [
            HitterGame.from_batting_stats(
                hitter,
                self.game_id,
                self.away_team) for hitter in player_statistics.away_batting]
        home_batters = [
            HitterGame.from_batting_stats(
                hitter,
                self.game_id,
                self.home_team) for hitter in player_statistics.home_batting]
        return home_batters + away_batters


class HitterGame():
    def __init__(self, ab, bb, bo, hbp, h, so, sac, player_id, game_id, team):
        self.ab = ab
        self.bb = bb
        self.bo = bo
        self.hbp = hbp
        self.h = h
        self.so = so
        self.sac = sac
        self.player_id = player_id
        self.game_id = game_id
        self.team = team
        self.pa = ab + bb + hbp + sac

    def __repr__(self):
        return '{}-{}'.format(self.player_id, self.game.game_id)

    def from_batting_stats(batting_stats, game_id, team):
        return HitterGame(
            ab=batting_stats.ab,
            bb=batting_stats.bb,
            bo=batting_stats.bo if hasattr(
                batting_stats, 'bo') else 9,
            hbp=batting_stats.hbp,
            h=batting_stats.hbp,
            so=batting_stats.so,
            sac=batting_stats.sac + batting_stats.sf,
            player_id=batting_stats.id,
            game_id=game_id,
            team=team)
