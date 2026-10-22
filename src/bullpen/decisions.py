"""Eligible decision points and the candidate relievers at each one.

A decision point is a mid-game pitching change (a reliever entering) where:
  * inning in cfg.decision_innings (7-9), in a game scheduled for 9 innings,
  * regular season, game not ending in a tie,
  * fielding-team margin in [margin_min, margin_max] at the moment of entry,
  * at least cfg.min_available relievers pass the availability rule.
Restricting to actual pitching changes means the team has already decided to
use *a* reliever; the counterfactual is only *which* one (A: best now,
B: second-best now and best held back).
"""
from __future__ import annotations

import pandas as pd

from .availability import UsageIndex, unavailable_reasons
from .config import Config
from .ratings import batter_pa_asof, ratings_asof


def build_decisions(apps, states, daily, params, cfg: Config):
    funnel = {}
    rel = apps[(apps["is_starter"] == 0) & apps["split"].isin(["train", "validation", "test"])]
    funnel["relief_appearances"] = len(rel)
    rel = rel[rel["entry_inning"].isin(cfg.decision_innings)];          funnel["innings_7_9"] = len(rel)
    rel = rel[(rel["sched_innings"] == 9) & (~rel["is_tie"])];        funnel["nine_inning_no_tie"] = len(rel)
    rel = rel[rel["entry_margin"].between(cfg.margin_min, cfg.margin_max)]
    funnel["close_margin"] = len(rel)

    idx = UsageIndex(apps)
    used_by_game = {k: g.sort_values("app_order")[["pitcher", "app_order"]].values.tolist()
                    for k, g in apps.groupby(["gid", "fld_team"])}

    rows = []
    for dp in rel.itertuples(index=False):
        d = dp.date.toordinal()
        used = {p for p, o in used_by_game[(dp.gid, dp.fld_team)] if o < dp.app_order}
        pool = idx.pool(dp.fld_team, d, cfg.pool_lookback_days) | {dp.pitcher}
        dec_id = f"{dp.gid}_{dp.fld_team}_{dp.app_order}"
        for p in pool:
            u = idx.usage(p, d, dp.game_num, cfg.starter_exclusion_days)
            if u["started_recently"] and p != dp.pitcher:
                continue
            reasons = unavailable_reasons(u, p in used, cfg)
            rows.append(dict(decision_id=dec_id, gid=dp.gid, fld_team=dp.fld_team, season=dp.season,
                             date=dp.date, pitcher=p, is_actual=int(p == dp.pitcher),
                             available=int(not reasons), unavail_reasons="|".join(reasons), **u))
    cand = pd.DataFrame(rows)
    if cand.empty:
        raise ValueError("No candidate relievers found; check inputs and config.")

    cand["batting_pa"] = batter_pa_asof(states, cand)
    cand = cand[(cand["batting_pa"] < cfg.position_player_min_pa) | (cand["is_actual"] == 1)]
    r = ratings_asof(cand, daily, params, cfg)
    cand = cand.merge(r, on=["pitcher", "season", "date"], how="left")

    av = cand[cand["available"] == 1].sort_values(["decision_id", "rating", "pitcher"])
    av["avail_rank"] = av.groupby("decision_id").cumcount() + 1
    cand = cand.merge(av[["decision_id", "pitcher", "avail_rank"]], on=["decision_id", "pitcher"], how="left")
    n_av = av.groupby("decision_id").size().rename("n_available")

    best = av[av["avail_rank"] == 1].set_index("decision_id")
    alt = av[av["avail_rank"] == 2].set_index("decision_id")
    act = cand[cand["is_actual"] == 1].set_index("decision_id")

    rel = rel.assign(decision_id=rel["gid"] + "_" + rel["fld_team"] + "_" + rel["app_order"].astype(str))
    dec = rel.set_index("decision_id").join(n_av).rename(columns={"pitcher": "actual_pitcher"})
    dec["n_available"] = dec["n_available"].fillna(0).astype(int)
    dec["best_pitcher"] = best["pitcher"];       dec["best_rating"] = best["rating"]
    dec["best_runs_saved_per100"] = best["runs_saved_per100"]
    dec["alt_pitcher"] = alt["pitcher"];         dec["alt_rating"] = alt["rating"]
    dec["alt_runs_saved_per100"] = alt["runs_saved_per100"]
    dec["rating_gap_per100"] = dec["best_runs_saved_per100"] - dec["alt_runs_saved_per100"]
    dec["actual_rating"] = act["rating"]
    dec["actual_available_by_rule"] = act["available"]
    dec["actual_avail_rank"] = act["avail_rank"]
    dec["actual_choice"] = pd.cut(dec["actual_avail_rank"].fillna(99), [0, 1, 2, 98, 100],
                                  labels=["best", "second", "other_available", "rule_unavailable"])
    dec["n_pool"] = cand.groupby("decision_id").size()

    dec = dec[dec["n_available"] >= cfg.min_available]
    funnel["eligible_decisions"] = len(dec)
    cand = cand[cand["decision_id"].isin(dec.index)]
    return dec.reset_index(), cand.reset_index(drop=True), funnel
