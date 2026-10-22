"""Bullpen pool and the transparent rest rule used as an availability proxy.

Pool (who counts as a reliever for team T on date d):
  * made >= 1 relief appearance for T in [d - pool_lookback_days, d - 1], and
  * did not start a game for T in [d - starter_exclusion_days, d], and
  * is not a position player (< position_player_min_pa batting PA, prior season + to date).
The pitcher who actually entered is always added to his own decision's pool.

Availability (pool member is unavailable if ANY rule fires):
  back_to_back     pitched on both d-1 and d-2
  heavy_prev_day   faced >= heavy_bf_prev_day batters on d-1
  workload_4d      more than max_apps_prev4 appearances in d-4 .. d-1
  pitched_today    pitched in an earlier game the same day (doubleheader)
  used_in_game     already pitched earlier in this game
This is a proxy. Public data cannot show injuries, illness, warm-up status or
the manager's own fatigue judgement.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict

import numpy as np
import pandas as pd

from .config import Config


class UsageIndex:
    def __init__(self, apps: pd.DataFrame):
        a = apps.assign(d=apps["date"].map(pd.Timestamp.toordinal)).sort_values(["d", "game_num"])
        self.by_pitcher = defaultdict(list)                       # pitcher -> [(d, gnum, bf, starter, team)]
        for r in a[["pitcher", "d", "game_num", "bf", "is_starter", "fld_team"]].itertuples(index=False):
            self.by_pitcher[r.pitcher].append((r.d, r.game_num, r.bf, r.is_starter, r.fld_team))
        self.days = {p: [x[0] for x in v] for p, v in self.by_pitcher.items()}
        rel = a[a["is_starter"] == 0]
        self.team_relief = {}
        for t, grp in rel.groupby("fld_team"):
            self.team_relief[t] = (grp["d"].to_numpy(), grp["pitcher"].to_numpy())

    def pool(self, team: str, d: int, lookback: int) -> set:
        if team not in self.team_relief:
            return set()
        days, pits = self.team_relief[team]
        lo, hi = np.searchsorted(days, d - lookback, "left"), np.searchsorted(days, d - 1, "right")
        return set(pits[lo:hi])

    def usage(self, pitcher: str, d: int, game_num: int, starter_days: int) -> dict:
        hist, days = self.by_pitcher.get(pitcher, []), self.days.get(pitcher, [])
        lo, hi = bisect_left(days, d - max(4, starter_days)), bisect_right(days, d)
        f = dict(pitched_d1=0, pitched_d2=0, bf_d1=0, apps_prev4=0,
                 pitched_earlier_today=0, started_recently=0)
        for (day, gnum, bf, starter, _team) in hist[lo:hi]:
            if day == d and gnum >= game_num:            # this game or later: unknown at decision time
                continue
            if day == d:
                f["pitched_earlier_today"] = 1
            if day == d - 1:
                f["pitched_d1"] = 1
                f["bf_d1"] += bf
            if day == d - 2:
                f["pitched_d2"] = 1
            if d - 4 <= day <= d - 1:
                f["apps_prev4"] += 1
            if starter and day >= d - starter_days:
                f["started_recently"] = 1
        return f


def unavailable_reasons(u: dict, used_in_game: bool, cfg: Config) -> list[str]:
    r = []
    if used_in_game:
        r.append("used_in_game")
    if cfg.rule_back_to_back and u["pitched_d1"] and u["pitched_d2"]:
        r.append("back_to_back")
    if cfg.heavy_bf_prev_day and u["bf_d1"] >= cfg.heavy_bf_prev_day:
        r.append("heavy_prev_day")
    if u["apps_prev4"] > cfg.max_apps_prev4:
        r.append("workload_4d")
    if cfg.rule_pitched_earlier_today and u["pitched_earlier_today"]:
        r.append("pitched_today")
    return r
