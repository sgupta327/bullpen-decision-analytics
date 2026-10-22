"""Reliever quality: run prevention per batter faced, shrunk toward league average.

1. A 24-state run-expectancy (RE) matrix is estimated on TRAINING seasons only.
2. Every event gets RE24 = runs + RE(next state) - RE(current state), charged to
   the pitcher on the mound (positive = runs allowed above expectation).
3. A pitcher's rating on date d uses only events strictly before d:
       S = season-to-date RE24 + w1 * last season + w2 * two seasons ago
       N = matching batters faced
       rating = (S + k * mu) / (N + k)
   mu  = league RE24 per BF for relievers (training seasons)
   k   = shrinkage strength in pseudo-batters, from a method-of-moments fit on
         training reliever-seasons (k = within-pitcher variance / between-pitcher variance)
Lower rating = better run prevention.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def re_matrix(states: pd.DataFrame, cfg: Config) -> pd.Series:
    tr = states[(states["split"] == "train") & (states["inning"] <= 8)
                & (states["half_truncated"] == 0)]
    re = tr.groupby("state24")["runs_rest_half"].mean()
    missing = 24 - len(re)
    if missing:
        raise ValueError(f"RE matrix is missing {missing} base-out states; not enough training data.")
    return re


def add_re24(states: pd.DataFrame, re: pd.Series) -> pd.DataFrame:
    df = states
    re_start = df["state24"].map(re).values
    nxt = df.groupby("half_id", sort=False)["state24"].shift(-1)
    re_end = nxt.map(re).fillna(0.0).values           # end of half-inning -> 0
    df["re_start"] = re_start
    df["re24"] = df["runs"].values + re_end - re_start
    # Last event of a walk-off half has no well-defined end state -> drop from ratings.
    is_last = nxt.isna().values
    df.loc[is_last & (df["half_truncated"].values == 1), "re24"] = np.nan
    return df


def tag_relief_rows(states: pd.DataFrame, apps: pd.DataFrame) -> pd.DataFrame:
    key = apps.set_index(["gid", "fld_team", "pitcher"])["is_starter"]
    idx = pd.MultiIndex.from_frame(states[["gid", "fld_team", "pitcher"]])
    states["pitcher_is_starter"] = key.reindex(idx).values.astype(int)
    return states


def fit_shrinkage(states: pd.DataFrame, cfg: Config) -> dict:
    rel = states[(states["split"] == "train") & (states["pitcher_is_starter"] == 0)
                 & states["re24"].notna()]
    total_bf = rel["is_pa"].sum()
    mu = rel["re24"].sum() / total_bf
    sigma2 = (rel["re24"] ** 2).sum() / total_bf - mu ** 2      # per-BF noise variance
    ps = rel.groupby(["pitcher", "season"]).agg(s=("re24", "sum"), n=("is_pa", "sum"))
    ps = ps[ps["n"] >= 50]
    if cfg.shrink_k is not None:
        k, tau2 = float(cfg.shrink_k), float("nan")
    elif len(ps) < 10:
        k, tau2 = float(cfg.shrink_k_bounds[1]), float("nan")
    else:
        r = ps["s"] / ps["n"]
        obs = np.average((r - mu) ** 2, weights=ps["n"])
        tau2 = obs - sigma2 * len(ps) / ps["n"].sum()
        k = sigma2 / tau2 if tau2 > 0 else cfg.shrink_k_bounds[1]
    lo, hi = cfg.shrink_k_bounds
    return {"mu": float(mu), "sigma2_per_bf": float(sigma2), "tau2_between": float(tau2),
            "k": float(np.clip(k, lo, hi)), "n_pitcher_seasons": int(len(ps))}


def daily_totals(states: pd.DataFrame) -> pd.DataFrame:
    s = states[states["re24"].notna()]
    return (s.groupby(["pitcher", "season", "date"])
             .agg(re24=("re24", "sum"), bf=("is_pa", "sum")).reset_index())


def ratings_asof(queries: pd.DataFrame, daily: pd.DataFrame, params: dict,
                 cfg: Config) -> pd.DataFrame:
    """queries: columns pitcher, season, date. Uses data strictly before `date`."""
    q = queries[["pitcher", "season", "date"]].drop_duplicates().copy()
    d = daily.sort_values(["pitcher", "season", "date"]).copy()
    d["cum_re24"] = d.groupby(["pitcher", "season"])["re24"].cumsum()
    d["cum_bf"] = d.groupby(["pitcher", "season"])["bf"].cumsum()
    cur = pd.merge_asof(q.sort_values("date"), d[["pitcher", "season", "date", "cum_re24", "cum_bf"]]
                        .sort_values("date"), on="date", by=["pitcher", "season"],
                        allow_exact_matches=False, direction="backward")
    cur[["cum_re24", "cum_bf"]] = cur[["cum_re24", "cum_bf"]].fillna(0.0)

    tot = daily.groupby(["pitcher", "season"])[["re24", "bf"]].sum()
    S, N = cur["cum_re24"].values.copy(), cur["cum_bf"].values.copy()
    for lag, w in enumerate(cfg.prior_season_weights, start=1):
        idx = pd.MultiIndex.from_arrays([cur["pitcher"], cur["season"] - lag])
        prev = tot.reindex(idx).fillna(0.0)
        S += w * prev["re24"].values
        N += w * prev["bf"].values
    k, mu = params["k"], params["mu"]
    cur["rating"] = (S + k * mu) / (N + k)
    cur["rating_eff_bf"] = N
    cur["runs_saved_per100"] = (mu - cur["rating"]) * 100   # + = better than average
    return cur[["pitcher", "season", "date", "rating", "rating_eff_bf", "runs_saved_per100"]]


def batter_pa_asof(states: pd.DataFrame, queries: pd.DataFrame) -> pd.Series:
    """Batting PA (prior season + season-to-date) for the position-player filter."""
    if "batter" not in states:
        return pd.Series(0, index=queries.index)
    b = (states[states["is_pa"] == 1].groupby(["batter", "season", "date"]).size()
         .rename("pa").reset_index().rename(columns={"batter": "pitcher"}))
    b = b.sort_values(["pitcher", "season", "date"])
    b["cum"] = b.groupby(["pitcher", "season"])["pa"].cumsum()
    q = queries[["pitcher", "season", "date"]].reset_index()
    cur = pd.merge_asof(q.sort_values("date"), b[["pitcher", "season", "date", "cum"]].sort_values("date"),
                        on="date", by=["pitcher", "season"], allow_exact_matches=False)
    tot = b.groupby(["pitcher", "season"])["pa"].sum()
    prev = tot.reindex(pd.MultiIndex.from_arrays([cur["pitcher"], cur["season"] - 1])).fillna(0).values
    cur["pa"] = cur["cum"].fillna(0).values + prev
    return cur.set_index("index")["pa"].reindex(queries.index)
