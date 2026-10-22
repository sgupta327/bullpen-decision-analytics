"""Synthetic play-by-play in Retrosheet plays.csv layout, for tests only.

It mimics the columns the pipeline reads (gid, inning, top_bot, batteam,
pitteam, batter, pitcher, outs_pre, br1_pre..br3_pre, runs, pa). Outcomes are
crude; the point is to exercise the code path, not to resemble MLB.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

TEAMS = ["ANA", "BOS", "CHN", "DET", "HOU", "NYA"]


def _team(code, rng):
    return {
        "code": code,
        "starters": [f"{code.lower()}s{i}" for i in range(5)],
        "relievers": {f"{code.lower()}r{i}": rng.normal(0, 0.04) for i in range(8)},
        "batters": [f"{code.lower()}b{i}" for i in range(9)],
        "position_pitcher": f"{code.lower()}b8",   # a position player who sometimes pitches
        "next_starter": 0, "next_batter": 0,
    }


def simulate_season(year, rng, n_days=45, dh_every=9):
    teams = {c: _team(c, rng) for c in TEAMS}
    last_used = {}                                  # pitcher -> list of date ordinals
    rows = []
    start = dt.date(year, 4, 1)
    for day in range(n_days):
        date = start + dt.timedelta(days=day)
        order = list(rng.permutation(TEAMS))
        matchups = [(order[i], order[i + 1]) for i in range(0, 6, 2)]
        games = [(h, v, 0) for v, h in matchups]
        if day % dh_every == 3:                    # one doubleheader
            h, v, _ = games[0]
            games[0] = (h, v, 1)
            games.append((h, v, 2))
        for home, vis, gnum in games:
            gid = f"{home}{date:%Y%m%d}{gnum}"
            rows += _game(gid, teams[home], teams[vis], date.toordinal(), last_used, rng)
    return pd.DataFrame(rows)


def _pick_reliever(team, d, used, margin, last_used, rng):
    def rested(p):
        u = last_used.get(p, [])
        return p not in used and not ((d - 1) in u and (d - 2) in u) and d not in u
    pool = [p for p in team["relievers"] if rested(p)]
    if abs(margin) >= 7 and rng.random() < 0.3:
        return team["position_pitcher"]
    if not pool:
        pool = [p for p in team["relievers"] if p not in used] or list(team["relievers"])
    if abs(margin) <= 3 and rng.random() < 0.6:
        return min(pool, key=lambda p: team["relievers"][p])
    return pool[rng.integers(len(pool))]


def _game(gid, home, vis, d, last_used, rng):
    rows, score = [], {0: 0, 1: 0}
    fld = {0: home, 1: vis}                         # keyed by bat_home: who is fielding
    bat = {0: vis, 1: home}
    cur = {}
    used = {0: set(), 1: set()}
    for t, key in ((home, 0), (vis, 1)):
        p = t["starters"][t["next_starter"] % 5]; t["next_starter"] += 1
        cur[key] = p; used[key].add(p)
    inning = 1
    while True:
        for bh in (0, 1):
            if bh == 1 and inning >= 9 and score[1] > score[0]:
                return _finish(rows, d, last_used)
            ft, bt = fld[bh], bat[bh]
            margin = score[1 - bh] - score[bh]
            if inning >= 7 and (inning > 7 or rng.random() < 0.7):
                p = _pick_reliever(ft, d, used[bh], margin, last_used, rng)
                cur[bh] = p; used[bh].add(p)
            outs, bases = 0, [None, None, None]
            if inning >= 10:
                bases[1] = bt["batters"][0]
            while outs < 3:
                if (outs >= 1 and any(bases) and inning >= 7 and abs(margin) <= 2
                        and rng.random() < 0.08):
                    p = _pick_reliever(ft, d, used[bh], margin, last_used, rng)
                    cur[bh] = p; used[bh].add(p)
                p = cur[bh]
                talent = ft["relievers"].get(p, 0.0) + (0.12 if p == ft["position_pitcher"] else 0)
                batter = bt["batters"][bt["next_batter"] % 9]; bt["next_batter"] += 1
                row = dict(gid=gid, inning=inning, top_bot=bh, batteam=bt["code"], pitteam=ft["code"],
                           batter=batter, pitcher=p, outs_pre=outs,
                           br1_pre=bases[0] or "", br2_pre=bases[1] or "", br3_pre=bases[2] or "", pa=1)
                u = rng.random()
                p_out = 0.68 - talent
                runs = 0
                if u < p_out:
                    outs += 1
                    if outs < 3 and any(bases) and rng.random() < 0.3:   # productive out
                        runs = int(bool(bases[2]))
                        bases = [None, bases[0], bases[1]]
                elif u < p_out + 0.10:                                  # walk
                    b1, b2, b3 = bases
                    if b1 and b2 and b3:
                        runs, bases = 1, [batter, b1, b2]
                    elif b1 and b2:
                        bases = [batter, b1, b2]
                    elif b1:
                        bases = [batter, b1, b3]
                    else:
                        bases = [batter, b2, b3]
                elif u < p_out + 0.24:                                  # single
                    runs = int(bool(bases[2])) + int(bool(bases[1]))
                    bases = [batter, None, bases[0]]
                elif u < p_out + 0.29:                                  # double
                    r1_holds = bool(bases[0]) and rng.random() < 0.5
                    runs = sum(bool(b) for b in bases[1:]) + int(bool(bases[0]) and not r1_holds)
                    bases = [None, batter, bases[0] if r1_holds else None]
                elif u < p_out + 0.30:                                  # triple
                    runs = sum(bool(b) for b in bases)
                    bases = [None, None, batter]
                else:                                                   # home run
                    runs = 1 + sum(bool(b) for b in bases)
                    bases = [None, None, None]
                row["runs"] = runs
                rows.append(row)
                score[bh] += runs
                if bh == 1 and inning >= 9 and score[1] > score[0]:
                    return _finish(rows, d, last_used)
        if inning >= 9 and score[0] != score[1]:
            return _finish(rows, d, last_used)
        inning += 1


def _finish(rows, d, last_used):
    gid = rows[0]["gid"]
    for p in {r["pitcher"] for r in rows if r["gid"] == gid}:
        last_used.setdefault(p, []).append(d)
    return rows


def write_synthetic(raw_dir, seasons, seed=7):
    rng = np.random.default_rng(seed)
    for y in seasons:
        df = simulate_season(y, rng)
        out = Path(raw_dir) / str(y)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "plays.csv", index=False)
