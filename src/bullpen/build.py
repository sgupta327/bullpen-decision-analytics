"""Build the reproducible game-state dataset.

    python -m bullpen.build --config configs/default.json

Outputs (in cfg.out_dir):
  states        one row per event state (all innings), features + targets
  decisions     one row per eligible decision point, with best/alt reliever
  candidates    one row per (decision point, bullpen pitcher), availability + rating
  games         one row per game
  re_matrix.csv, rating_params.json, manifest.json
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .availability import UsageIndex
from .config import Config
from .decisions import build_decisions
from .io_retrosheet import load_seasons
from .ratings import add_re24, daily_totals, fit_shrinkage, ratings_asof, re_matrix, tag_relief_rows
from .states import build_appearances, build_states

STATE_COLS = [
    "gid", "season", "split", "date", "game_num", "row_idx", "half_id", "inning", "bat_home",
    "bat_team", "fld_team", "pitcher", "is_pa", "outs_pre", "br1", "br2", "br3", "base_state",
    "state24", "bat_score_pre", "fld_score_pre", "fld_margin", "fld_is_home", "sched_innings",
    "extra_runner_era", "runs", "re_start", "re24",
    # pitcher features (known before the event)
    "pitcher_is_starter", "pitcher_rating", "pitcher_runs_saved_per100", "pitcher_rating_eff_bf",
    "pitcher_bf_in_game", "pitched_d1", "pitched_d2", "bf_d1", "apps_prev4",
    # targets
    "runs_rest_half", "scored_rest_half", "half_truncated", "fld_win", "fld_runs_allowed_rest_game",
    "is_tie",
]


def write_table(df: pd.DataFrame, path: Path) -> str:
    try:
        import pyarrow  # noqa: F401
        out = path.with_suffix(".parquet")
        df.to_parquet(out, index=False)
    except ImportError:
        out = path.with_suffix(".csv.gz")
        df.to_csv(out, index=False)
    return str(out)


def add_pitcher_features(states, apps, daily, params, cfg):
    r = ratings_asof(states, daily, params, cfg)
    states = states.merge(r.rename(columns={"rating": "pitcher_rating",
                                            "runs_saved_per100": "pitcher_runs_saved_per100",
                                            "rating_eff_bf": "pitcher_rating_eff_bf"}),
                          on=["pitcher", "season", "date"], how="left")
    states = states.sort_values("row_idx").reset_index(drop=True)
    states["pitcher_bf_in_game"] = (states.groupby(["gid", "pitcher"])["is_pa"].cumsum()
                                    - states["is_pa"])
    idx = UsageIndex(apps)
    keys = apps[["gid", "pitcher", "date", "game_num"]].drop_duplicates(["gid", "pitcher"])
    u = pd.DataFrame([idx.usage(k.pitcher, k.date.toordinal(), k.game_num, cfg.starter_exclusion_days)
                      for k in keys.itertuples(index=False)], index=keys.index)
    keys = pd.concat([keys[["gid", "pitcher"]], u], axis=1)
    return states.merge(keys[["gid", "pitcher", "pitched_d1", "pitched_d2", "bf_d1", "apps_prev4"]],
                        on=["gid", "pitcher"], how="left")


def run(cfg: Config) -> dict:
    t0 = time.time()
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    plays, files = load_seasons(cfg.all_seasons, cfg.raw_dir, cfg.regular_season_only)
    states, games = build_states(plays, cfg)
    apps = build_appearances(states)
    states = tag_relief_rows(states, apps)

    re = re_matrix(states, cfg)                       # training seasons only
    states = add_re24(states, re)
    params = fit_shrinkage(states, cfg)               # training seasons only
    daily = daily_totals(states)

    states = add_pitcher_features(states, apps, daily, params, cfg)
    decisions, candidates, funnel = build_decisions(apps, states, daily, params, cfg)

    model_states = states[states["split"].isin(["train", "validation", "test"])][STATE_COLS]
    paths = {
        "states": write_table(model_states, out / "states"),
        "decisions": write_table(decisions, out / "decisions"),
        "candidates": write_table(candidates, out / "candidates"),
        "games": write_table(games, out / "games"),
    }
    re.rename("expected_runs").to_csv(out / "re_matrix.csv")
    (out / "rating_params.json").write_text(json.dumps(params, indent=2))

    by_split = lambda df: df.groupby("split").size().to_dict() if "split" in df else {}
    manifest = {
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(time.time() - t0, 1),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "pandas": pd.__version__, "numpy": np.__version__,
        "config": cfg.to_dict(), "input_files": files, "outputs": paths,
        "rows": {"states": by_split(model_states), "decisions": by_split(decisions),
                 "candidates": len(candidates), "games": by_split(games)},
        "decision_funnel": funnel,
        "actual_choice_share": decisions["actual_choice"].value_counts(normalize=True).round(4).to_dict(),
        "rating_params": params,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="JSON config (default: built-in defaults)")
    args = ap.parse_args(argv)
    m = run(Config.from_json(args.config))
    print(json.dumps({k: m[k] for k in ("rows", "decision_funnel", "actual_choice_share",
                                        "rating_params", "outputs")}, indent=2, default=str))


if __name__ == "__main__":
    main()
