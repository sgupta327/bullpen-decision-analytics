"""Turn event rows into game states with score context and outcome targets.

Each row is the state *before* an event: inning, half, outs, runners, score.
Targets are attached from what happened afterwards in the same game.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def build_states(plays: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = plays.copy()
    df["row_idx"] = np.arange(len(df))
    df["half_id"] = df["gid"] + "_" + df["inning"].astype(str) + "_" + df["bat_home"].astype(str)

    # ---- score before each event -------------------------------------------
    home_r = np.where(df["bat_home"] == 1, df["runs"], 0)
    vis_r = np.where(df["bat_home"] == 0, df["runs"], 0)
    g = df.groupby("gid", sort=False)
    df["home_score_pre"] = pd.Series(home_r).groupby(df["gid"]).cumsum().values - home_r
    df["vis_score_pre"] = pd.Series(vis_r).groupby(df["gid"]).cumsum().values - vis_r

    extra = {"sched_from_gameinfo": ("sched_innings", "first")} if "sched_innings" in df else {}
    games = g.agg(season=("season", "first"), date=("date", "first"), **extra,
                  game_num=("game_num", "first"), home_team=("home_team", "first"),
                  max_inning=("inning", "max")).reset_index()
    finals = pd.DataFrame({"gid": df["gid"], "h": home_r, "v": vis_r}).groupby("gid").sum()
    games = games.merge(finals.rename(columns={"h": "home_final", "v": "vis_final"}),
                        left_on="gid", right_index=True)
    # "Home" = the team batting last. In a few games (neutral sites, 2015 BAL@TB,
    # 2020 relocations) the park's team batted first; game id prefix is the park.
    games = games.rename(columns={"home_team": "park_team"})
    vis = df.loc[df["bat_home"] == 0].groupby("gid")["bat_team"].first()
    home = df.loc[df["bat_home"] == 1].groupby("gid")["bat_team"].first()
    games["vis_team"] = games["gid"].map(vis)
    games["home_team"] = games["gid"].map(home)
    games["is_tie"] = games["home_final"] == games["vis_final"]
    games["home_win"] = (games["home_final"] > games["vis_final"]).astype(int)
    # Scheduled length: gameinfo.csv when present, else the 2020-21 doubleheader rule.
    rule = np.where(games["season"].isin(cfg.seven_inning_dh_seasons)
                    & games["game_num"].isin([1, 2]), 7, 9)
    sched = games.pop("sched_from_gameinfo") if "sched_from_gameinfo" in games else None
    games["sched_innings"] = rule if sched is None else sched.fillna(pd.Series(rule)).astype(int)
    games["split"] = games["season"].map(cfg.split_of)

    df = df.drop(columns=["sched_innings"], errors="ignore")
    df = df.merge(games[["gid", "home_final", "vis_final", "home_win", "is_tie",
                         "sched_innings"]], on="gid", how="left")

    bh = df["bat_home"] == 1
    df["bat_score_pre"] = np.where(bh, df["home_score_pre"], df["vis_score_pre"])
    df["fld_score_pre"] = np.where(bh, df["vis_score_pre"], df["home_score_pre"])
    df["fld_margin"] = df["fld_score_pre"] - df["bat_score_pre"]  # + = fielding team leads
    df["fld_is_home"] = 1 - df["bat_home"]

    # ---- targets ------------------------------------------------------------
    # runs the batting team scores from this state to the end of the half-inning
    df["runs_rest_half"] = (df.iloc[::-1].groupby("half_id", sort=False)["runs"].cumsum()
                            .iloc[::-1].values)
    df["scored_rest_half"] = (df["runs_rest_half"] > 0).astype(int)
    fld_final = np.where(bh, df["vis_final"], df["home_final"])
    bat_final = np.where(bh, df["home_final"], df["vis_final"])
    df["fld_win"] = (fld_final > bat_final).astype(int)
    df["fld_runs_allowed_rest_game"] = bat_final - df["bat_score_pre"]

    # Half-innings cut short by a walk-off: their remaining runs are censored.
    last_half = df.groupby("gid")["half_id"].transform("last")
    df["half_truncated"] = ((df["half_id"] == last_half) & bh &
                            (df["home_final"] > df["vis_final"])).astype(int)

    df["base_state"] = df["br1"].astype(str) + df["br2"].astype(str) + df["br3"].astype(str)
    df["state24"] = df["outs_pre"].astype(str) + "_" + df["base_state"]
    df["extra_runner_era"] = (df["season"] >= 2020).astype(int)  # runner on 2nd in extras
    df["split"] = df["season"].map(cfg.split_of)
    return df, games


def build_appearances(states: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, team, pitcher): entry state, batters faced, starter flag."""
    first = states.sort_values("row_idx").drop_duplicates(["gid", "fld_team", "pitcher"])
    apps = first[["gid", "season", "date", "game_num", "fld_team", "pitcher", "row_idx",
                  "inning", "outs_pre", "base_state", "fld_margin", "fld_is_home",
                  "sched_innings", "is_tie", "fld_win", "split"]].rename(columns={
                      "row_idx": "entry_row", "inning": "entry_inning",
                      "outs_pre": "entry_outs", "base_state": "entry_bases",
                      "fld_margin": "entry_margin"})
    bf = states.groupby(["gid", "fld_team", "pitcher"])["is_pa"].sum().rename("bf")
    apps = apps.merge(bf, left_on=["gid", "fld_team", "pitcher"], right_index=True)
    apps = apps.sort_values(["gid", "fld_team", "entry_row"])
    apps["app_order"] = apps.groupby(["gid", "fld_team"]).cumcount() + 1
    apps["is_starter"] = (apps["app_order"] == 1).astype(int)
    return apps.reset_index(drop=True)
