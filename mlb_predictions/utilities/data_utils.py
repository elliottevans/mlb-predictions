import pandas as pd
from io import StringIO
import jinja2
from dataclasses import dataclass
from urllib3.exceptions import HTTPError
from typing import List, Dict, Union, Tuple, Optional
import sys
import os
import numpy as np
from datetime import date, datetime, timedelta
import warnings
import statsapi
from pybaseball import playerid_lookup, cache
import requests
from bs4 import BeautifulSoup
from multiprocess import Pool

warnings.simplefilter(action="ignore", category=FutureWarning)
cache.enable()

DATE_FORMAT = "%Y-%m-%d"


def validate_datestring(date_text: Optional[str]) -> date:
    try:
        assert date_text
        return datetime.strptime(date_text, DATE_FORMAT).date()
    except (AssertionError, ValueError) as ex:
        raise ValueError("Incorrect data format, should be YYYY-MM-DD") from ex


def sanitize_date_range(start_dt: Optional[str], end_dt: Optional[str]):
    # If no dates are supplied, assume they want yesterday's data
    # send a warning in case they wanted to specify
    if start_dt is None and end_dt is None:
        today = date.today()
        start_dt = str(today - timedelta(1))
        end_dt = str(today)

        print("start_dt", start_dt)
        print("end_dt", end_dt)

        print("Warning: no date range supplied, assuming yesterday's date.")

    # If only one date is supplied, assume they only want that day's stats
    # query in this case is from date 1 to date 1
    if start_dt is None:
        start_dt = end_dt
    if end_dt is None:
        end_dt = start_dt

    start_dt_date = validate_datestring(start_dt)
    end_dt_date = validate_datestring(end_dt)

    # If end date occurs before start date, swap them
    if end_dt_date < start_dt_date:
        start_dt_date, end_dt_date = end_dt_date, start_dt_date

    # Now that both dates are not None, make sure they are valid date strings
    return start_dt_date, end_dt_date


def get_hitting_soup(start_dt: date, end_dt: date) -> BeautifulSoup:
    # get most recent standings if date not specified
    # if((start_dt is None) or (end_dt is None)):
    #    print('Error: a date range needs to be specified')
    #    return None
    url = "http://www.baseball-reference.com/leagues/daily.cgi?user_team=&bust_cache=&type=b&lastndays=7&dates=fromandto&fromandto={}.{}&level=mlb&franch=&stat=&stat_value=0".format(
        start_dt, end_dt
    )
    s = requests.get(url).content
    # a workaround to avoid beautiful soup applying the wrong encoding
    s = str(s).encode()
    return BeautifulSoup(s, features="lxml")


def get_pitching_soup(start_dt, end_dt):
    # get most recent standings if date not specified
    if (start_dt is None) or (end_dt is None):
        print("Error: a date range needs to be specified")
        return None
    url = "http://www.baseball-reference.com/leagues/daily.cgi?user_team=&bust_cache=&type=p&lastndays=7&dates=fromandto&fromandto={}.{}&level=mlb&franch=&stat=&stat_value=0".format(
        start_dt, end_dt
    )
    s = requests.get(url).content
    # a workaround to avoid beautiful soup applying the wrong encoding
    s = str(s).encode()
    return BeautifulSoup(s, features="lxml")


def get_table(soup):
    table = soup.find_all("table")[0]
    data = []
    headings = [th.get_text() for th in table.find("tr").find_all("th")][1:]
    headings.append("mlbID")
    data.append(headings)
    table_body = table.find("tbody")
    rows = table_body.find_all("tr")
    for row in rows:
        cols = row.find_all("td")
        row_anchor = row.find("a")
        mlbid = (
            row_anchor["href"].split("mlb_ID=")[-1] if row_anchor else pd.NA
        )  # ID str or nan
        cols = [ele.text.strip() for ele in cols]
        cols.append(mlbid)
        data.append([ele for ele in cols])
    data = pd.DataFrame(data)
    data = data.rename(columns=data.iloc[0])
    data = data.reindex(data.index.drop(0))
    return data


def batting_stats_range(
    start_dt: Optional[str] = None, end_dt: Optional[str] = None
) -> pd.DataFrame:
    """
    Get all batting stats for a set time range. This can be the past week, the
    month of August, anything. Just supply the start and end date in YYYY-MM-DD
    format.
    """
    # make sure date inputs are valid
    start_dt_date, end_dt_date = sanitize_date_range(start_dt, end_dt)
    if start_dt_date.year < 2008:
        raise ValueError("Year must be 2008 or later")
    if end_dt_date.year < 2008:
        raise ValueError("Year must be 2008 or later")
    # retrieve html from baseball reference
    soup = get_hitting_soup(start_dt_date, end_dt_date)
    table = get_table(soup)
    table = table.dropna(how="all")  # drop if all columns are NA
    # scraped data is initially in string format.
    # convert the necessary columns to numeric.
    for column in [
        "Age",
        "#days",
        "G",
        "PA",
        "AB",
        "R",
        "H",
        "2B",
        "3B",
        "HR",
        "RBI",
        "BB",
        "IBB",
        "SO",
        "HBP",
        "SH",
        "SF",
        "GDP",
        "SB",
        "CS",
        "BA",
        "OBP",
        "SLG",
        "OPS",
        "mlbID",
    ]:
        # table[column] = table[column].astype('float')
        table[column] = pd.to_numeric(table[column])
        # table['column'] = table['column'].convert_objects(convert_numeric=True)
    table = table.drop("", 1)
    return table


def pitching_stats_range(start_dt=None, end_dt=None):
    """
    Get all pitching stats for a set time range. This can be the past week, the
    month of August, anything. Just supply the start and end date in YYYY-MM-DD
    format.
    """
    # ensure valid date strings, perform necessary processing for query
    start_dt_date, end_dt_date = sanitize_date_range(start_dt, end_dt)
    if start_dt_date.year < 2008:
        raise ValueError("Year must be 2008 or later")
    if end_dt_date.year < 2008:
        raise ValueError("Year must be 2008 or later")
    # retrieve html from baseball reference
    soup = get_pitching_soup(start_dt_date, end_dt_date)
    table = get_table(soup)
    table = table.dropna(how="all")  # drop if all columns are NA
    # fix some strange formatting for percentage columns
    table = table.replace("---%", np.nan)
    # make sure these are all numeric
    for column in [
        "Age",
        "#days",
        "G",
        "GS",
        "W",
        "L",
        "SV",
        "IP",
        "H",
        "R",
        "ER",
        "BB",
        "SO",
        "HR",
        "HBP",
        "ERA",
        "AB",
        "2B",
        "3B",
        "IBB",
        "GDP",
        "SF",
        "SB",
        "CS",
        "PO",
        "BF",
        "Pit",
        "WHIP",
        "BAbip",
        "SO9",
        "SO/W",
    ]:
        table[column] = pd.to_numeric(table[column])
    # convert str(xx%) values to float(0.XX) decimal values
    for column in ["Str", "StL", "StS", "GB/FB", "LD", "PU"]:
        table[column] = table[column].replace("%", "", regex=True).astype("float") / 100

    table = table.drop("", 1)
    return table


CLASS_REPR_TEMPLATE = """\
{{ name }}({% for attr, attr_value in vars.items() %}
    {{ attr }}: {{ attr_value }}{% if not loop.last %},{% endif %}{% endfor %}
)
"""


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def _player_search_list(last: str, first: str, year_filter: int, verbose: bool = False):
    cols = [
        "name_last",
        "name_first",
        "key_mlbam",
        "key_retro",
        "key_bbref",
        "key_fangraphs",
        "mlb_played_first",
        "mlb_played_last",
    ]

    if verbose:
        ids = playerid_lookup(last=last, first=first, fuzzy=True)
    else:
        with HiddenPrints():
            ids = playerid_lookup(last=last, first=first, fuzzy=True)

    ids = ids[ids.mlb_played_last != ""]

    # if we get multiple folks from a fuzzy search, get filter to
    # recent players
    if year_filter:
        ids = ids[ids.mlb_played_last >= year_filter - 5]
    ids = ids.head(1)

    if ids.empty:
        # empty search, just return row or None's
        ids = pd.DataFrame([None] * len(cols)).T
        ids.columns = cols

    return ids


def player_search_list(
    last=List[str],
    first=List[str],
    year_filter: int = None,
    n_jobs: int = 2,
    verbose: bool = False,
) -> pd.core.frame.DataFrame:
    """
    Lookup playerIDs (MLB AM, bbref, retrosheet, FG) for a list of players.

    Args:
        player_list: List of (last, first) tupels.

    Returns:
        pd.DataFrame: DataFrame of playerIDs, name, years played
    """

    with Pool(n_jobs) as p:
        _results = p.starmap(
            _player_search_list,
            zip(last, first, [year_filter] * len(last), [verbose] * len(last)),
        )

    results = pd.concat(_results)

    return results.reset_index(drop=True)


def impute_attrs(obj, lookup_func):
    """Helper function to impute all the attributes of an instance given
    a look up function.
    """
    attrs = list(vars(obj).values())
    non_null_attrs = [a for a in attrs if a is not None]

    if not non_null_attrs:
        raise ValueError("Must instantiate with at least one identifier!")

    if len(non_null_attrs) < len(attrs):
        # missing identifiers... impute them
        try:
            full_attrs = lookup_func(non_null_attrs[0])[0]
        except IndexError as e:
            raise IndexError(f"empty search for {non_null_attrs[0]}")

        for attr, value in full_attrs.items():
            obj.__setattr__(attr, value)


@dataclass
class HittingStats:
    basic: Tuple = (
        "Age",
        "G",
        "PA",
        "AB",
        "R",
        "H",
        "2B",
        "3B",
        "HR",
        "RBI",
        "BB",
        "IBB",
        "SO",
        "HBP",
        "SH",
        "SF",
        "GDP",
        "SB",
        "CS",
        "BA",
        "OBP",
        "SLG",
        "OPS",
    )


@dataclass
class PitchingStats:
    basic: Tuple = (
        "Age",
        "G",
        "GS",
        "W",
        "L",
        "SV",
        "IP",
        "H",
        "R",
        "ER",
        "BB",
        "SO",
        "HR",
        "HBP",
        "ERA",
        "AB",
        "2B",
        "3B",
        "IBB",
        "GDP",
        "SF",
        "SB",
        "CS",
        "PO",
        "BF",
        "Pit",
        "Str",
        "StL",
        "StS",
        "GB/FB",
        "LD",
        "PU",
        "WHIP",
        "BAbip",
        "SO9",
        "SO/W",
    )


@dataclass
class Player:
    id: int = None
    fullName: str = None
    firstName: str = None
    lastName: str = None
    primaryNumber: str = None
    currentTeam: Dict = None
    primaryPosition: Dict = None
    useName: str = None
    boxscoreName: str = None
    mlbDebutDate: str = None
    nameFirstLast: str = None
    firstLastName: str = None
    lastFirstName: str = None
    lastInitName: str = None
    initLastName: str = None
    fullFMLName: str = None
    fullLFMName: str = None

    def __post_init__(self):
        impute_attrs(self, lookup_func=statsapi.lookup_player)

    def __repr__(self):
        return jinja2.Template(CLASS_REPR_TEMPLATE).render(
            name=self.__class__.__name__, vars=vars(self)
        )


@dataclass
class Team:
    id: int = None
    name: str = None
    teamCode: str = None
    fileCode: str = None
    teamName: str = None
    locationName: str = None
    shortName: str = None

    def __post_init__(self):
        impute_attrs(self, lookup_func=statsapi.lookup_team)

    def __repr__(self):
        return jinja2.Template(CLASS_REPR_TEMPLATE).render(
            name=self.__class__.__name__, vars=vars(self)
        )

    def _get_roster_text(self, date: str, roster_type: Dict) -> str:
        """Helper function to get the raw roster text"""
        roster_text = statsapi.roster(
            self.id, date=date, rosterType=roster_type  # yyyy-mm-dd
        )
        return roster_text

    def _munge_roster(self, roster_text: str) -> pd.core.frame.DataFrame:
        """Helper function to munge the raw roster text into a dataframe"""
        munged_roster = (
            pd.read_csv(
                StringIO(roster_text),
                header=None,
                engine="python",
                delimiter=r"\s{2,}|TWP",
            )
            .fillna("TWP")
            .iloc[:, 1:]
            .drop_duplicates()
            .reset_index(drop=True)
            .rename(columns={1: "position", 2: "player"})
        )

        munged_roster["player"] = munged_roster.player.str.strip()

        return munged_roster

    def get_roster(
        self,
        date: str,
        roster_type: str = "active",
        as_dataframe: bool = False,
        n_jobs: int = 5,
        verbose: bool = False,
    ) -> Union[List[Player], pd.core.frame.DataFrame]:
        roster_types = RosterTypes()

        if not roster_types.is_valid_value(roster_type):
            raise ValueError(f"roster_type must be one of: {roster_types.get()}")

        _roster = statsapi.get(
            "team_roster", {"rosterType": roster_type, "date": date, "teamId": self.id}
        )

        roster = pd.DataFrame(
            [
                (
                    r["person"]["id"],
                    r["person"]["fullName"],
                    r["position"]["abbreviation"],
                )
                for r in _roster["roster"]
            ],
            columns=["key_mlbam", "player", "position"],
        )

        if as_dataframe:
            return roster
        else:
            return [Player(p) for p in roster.player]

    def get_roster_stats(
        self,
        date: str,
        stat_type: str,
        stat_end_date: str,
        stat_start_date: str = "2008-01-01",
        roster_type: str = "active",
        league_stats: pd.core.frame.DataFrame = None,
        n_jobs=10,
        verbose: bool = False,
    ) -> pd.core.frame.DataFrame:
        """Get stats for a team's roster on `date`. `stat_type` is either
        "hitting" or "pitching"
        """

        roster_df = self.get_roster(
            date=date,
            roster_type=roster_type,
            as_dataframe=True,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        keys = ["player", "key_mlbam"]

        if stat_type == "pitching":
            if league_stats is not None:
                stats = league_stats
            else:
                stats = pitching_stats_range(
                    start_dt=stat_start_date, end_dt=stat_end_date
                )

            cols_to_fetch = keys + [
                "Age",
                "G",
                "GS",
                "W",
                "L",
                "SV",
                "IP",
                "H",
                "R",
                "ER",
                "BB",
                "SO",
                "HR",
                "HBP",
                "ERA",
                "AB",
                "2B",
                "3B",
                "IBB",
                "GDP",
                "SF",
                "SB",
                "CS",
                "PO",
                "BF",
                "Pit",
                "Str",
                "StL",
                "StS",
                "GB/FB",
                "LD",
                "PU",
                "WHIP",
                "BAbip",
                "SO9",
                "SO/W",
            ]

        elif stat_type == "hitting":
            if league_stats is not None:
                stats = league_stats
            else:
                stats = batting_stats_range(
                    start_dt=stat_start_date, end_dt=stat_end_date
                )

            cols_to_fetch = keys + [
                "Age",
                "G",
                "PA",
                "AB",
                "R",
                "H",
                "2B",
                "3B",
                "HR",
                "RBI",
                "BB",
                "IBB",
                "SO",
                "HBP",
                "SH",
                "SF",
                "GDP",
                "SB",
                "CS",
                "BA",
                "OBP",
                "SLG",
                "OPS",
            ]
        else:
            raise ValueError("stat_type must be one of ('pitching', 'hitting')")

        stats = stats.rename(columns={"mlbID": "key_mlbam"}).astype({"key_mlbam": int})

        roster_stats = (
            roster_df.merge(stats, on="key_mlbam", how="left")
            .drop(columns="Name")
            .loc[:, cols_to_fetch]
        )

        assert len(roster_df) == len(roster_stats), (
            f"roster has {len(roster_df[msk])} rows, but roster stats"
            f" has {len(roster_stats)} rows!"
        )

        return roster_stats


@dataclass
class Game:
    game_id: int
    game_datetime: str = None
    game_date: str = None
    game_type: str = None
    status: str = None
    away_name: str = None
    home_name: str = None
    away_id: int = None
    home_id: int = None
    doubleheader: str = None
    game_num: int = None
    home_probable_pitcher: str = None
    away_probable_pitcher: str = None
    home_pitcher_note: str = None
    away_pitcher_note: str = None
    away_score: int = None
    home_score: int = None
    current_inning: int = None
    inning_state: str = None
    venue_id: int = None
    venue_name: int = None
    national_broadcasts: List[str] = None
    series_status: str = None
    winning_team: str = None
    losing_team: str = None
    winning_pitcher: str = None
    losing_pitcher: str = None
    save_pitcher: str = None
    summary: str = None

    def __repr__(self):
        return jinja2.Template(CLASS_REPR_TEMPLATE).render(
            name=self.__class__.__name__, vars=vars(self)
        )

    def get_away_team(self):
        _team = statsapi.lookup_team(self.away_id)[0]
        return Team(**_team)

    def get_home_team(self):
        _team = statsapi.lookup_team(self.home_id)[0]
        return Team(**_team)

    def _get_box_score(self, home_away: str) -> pd.core.frame.DataFrame:
        """Get a dataframe representation of the box score"""

        _home_away = "homeBatters" if home_away == "home" else "awayBatters"

        cols = [
            "personId",
            "name",
            "battingOrder",
            "substitution",
            "ab",
            "r",
            "h",
            "doubles",
            "triples",
            "hr",
            "rbi",
            "sb",
            "bb",
            "k",
        ]
        for i in range(3):
            while True:
                try:
                    raw = pd.DataFrame(
                        statsapi.boxscore_data(self.game_id)[_home_away]
                    )[cols]
                except HTTPError:
                    continue
                break

        msk_valid = (raw["personId"] != 0) & (raw["substitution"] == False)

        raw = raw[msk_valid].reset_index(drop=True).drop(columns=["substitution"])

        raw = raw.astype(
            {
                stat: int
                for stat in [
                    "ab",
                    "r",
                    "h",
                    "doubles",
                    "triples",
                    "hr",
                    "rbi",
                    "sb",
                    "bb",
                    "k",
                ]
            }
        )

        raw["singles"] = raw["h"] - raw["doubles"] - raw["triples"] - raw["hr"]

        raw["battingOrder"] = raw["battingOrder"].str[0].astype(int)

        raw["game_id"] = self.game_id

        box_score = raw.rename(columns={"personId": "player_id", "name": "lastName"})

        return box_score

    def get_away_box_score(self) -> pd.core.frame.DataFrame:
        return self._get_box_score("away")

    def get_home_box_score(self) -> pd.core.frame.DataFrame:
        return self._get_box_score("home")


@dataclass
class GameTypes:
    def __post_init__(self):
        self.valid_values = sorted(
            list(set(d["description"] for d in statsapi.meta("gameTypes")))
        )
        # alias -> valid_value
        self.key = {d["description"]: d["id"] for d in statsapi.meta("gameTypes")}

    def get(self):
        return sorted(list(self.key.keys()))

    def __repr__(self):
        return "\n".join(self.get())

    def is_valid_value(self, value):
        return value in self.get()


@dataclass
class GameStatuses:
    def __post_init__(self):
        self.valid_values = sorted(
            list(set(d["detailedState"] for d in statsapi.meta("gameStatus")))
        )

    def get(self):
        return self.valid_values

    def __repr__(self):
        return "\n".join(self.get())

    def is_valid_value(self, value):
        return value in self.get()


@dataclass
class RosterTypes:
    def __post_init__(self):
        _valid_values = statsapi.meta("rosterTypes")
        self.valid_values = sorted(
            [
                d["parameter"]
                for d in _valid_values
                if d["parameter"] not in ("coach", "gameday")
            ]
        )

    def get(self):
        return self.valid_values

    def __repr__(self):
        return "\n".join(self.get())

    def is_valid_value(self, value):
        return value in self.get()


@dataclass
class Schedule:
    year: Union[int, List[int]] = None
    date: Union[str, List[str]] = None

    def __post_init__(self):
        if not self.year and not self.date:
            raise ValueError("Either `years` or `dates` must be defined!")
        self._dates = [self.date] if not isinstance(self.date, list) else self.date
        self._years = [self.year] if not isinstance(self.year, list) else self.year

        # format the calendar dates
        self._dates = [
            pd.to_datetime(d).date().strftime("%m/%d/%Y")
            for d in self._dates
            if d is not None
        ]
        self._years = [y for y in self._years if y is not None]

        # valid calendar dates for each year
        self._year_dates = [
            date.date().strftime("%m/%d/%Y")
            for subyears in [
                pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="D")[:-1]
                for year in self._years
            ]
            for date in subyears
        ]

        if not self._year_dates:
            # _year_dates is empty, just force to be _dates
            self._year_dates = self._dates
        if not self._dates:
            # _dates is empty, just force to be _year_dates
            self._dates = self._year_dates

        self._valid_dates = list(set(self._year_dates).intersection(set(self._dates)))

    def _dates_are_consecutive(self, dates: List[str]):
        """Helper method to determine if string dates in a list are consecutive"""
        ordinal_dates = sorted([pd.to_datetime(d).toordinal() for d in dates])
        ordinal_dates = [d - min(ordinal_dates) for d in ordinal_dates]

        return ordinal_dates == list(range(len(ordinal_dates)))

    def _get_games(
        self,
        dates: List[str],
        game_type: Union[str, List[str]] = "regular",
        game_status: Union[str, List[str]] = "final",
    ) -> List[dict]:
        """Helper function to get raw list of dictionary games returned
        by MLBs API
        """

        # retry logic in case of stochastic api errors
        for i in range(3):
            while True:
                try:
                    if self._dates_are_consecutive(dates):
                        _games = statsapi.schedule(
                            start_date=min(dates), end_date=max(dates)
                        )

                    else:
                        _games = []
                        for d in dates:
                            _games.extend(statsapi.schedule(date=d))

                except HTTPError:
                    continue
                break

        if self._dates_are_consecutive(dates):
            _games = statsapi.schedule(start_date=min(dates), end_date=max(dates))

        else:
            _games = []
            for d in dates:
                _games.extend(statsapi.schedule(date=d))

        # munge the game types
        game_types = [game_type] if not isinstance(game_type, list) else game_type
        game_types = [GameTypes().key[gt] for gt in game_types]

        # munge the game statuses
        game_statuses = (
            [game_status] if not isinstance(game_status, list) else game_status
        )

        _games = [
            g
            for g in _games
            if g["game_type"] in game_types and g["status"] in game_statuses
        ]

        return _games

    def get_game_days(
        self,
        game_type: Union[str, List[str]] = "regular",
        game_status: Union[str, List[str]] = "final",
    ) -> List[str]:
        """Get days in which there are scheduled games"""

        game_types = GameTypes()
        if not game_types.is_valid_value(game_type):
            raise ValueError(f"game_type must be one of: {game_types.get()}")

        game_statuses = GameStatuses()
        if not game_statuses.is_valid_value(game_status):
            raise ValueError(f"game_status must be one of: {game_statuses.get()}")

        _games = self._get_games(
            dates=self._valid_dates, game_type=game_type, game_status=game_status
        )

        game_days = sorted(list(set([gd["game_date"] for gd in _games])))

        return game_days

    def get_games(
        self,
        game_type: Union[str, List[str]] = "Regular Season",
        game_status: Union[str, List[str]] = "Final",
    ) -> List[Game]:
        """Get days in which there are scheduled games"""

        game_types = GameTypes()

        if not game_types.is_valid_value(game_type):
            raise ValueError(f"game_type must be one of: {game_types.get()}")

        _games = self._get_games(
            dates=self._valid_dates, game_type=game_type, game_status=game_status
        )

        games = [Game(**g) for g in _games]

        return games
