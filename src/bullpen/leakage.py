"""Data-leakage checks for the Y1 models, run on the real processed dataset.

    python -m bullpen.leakage [--config configs/default.json] [--n 3000]

Each check recomputes something independently of the build code and compares:
  1. feature list contains no post-event / outcome columns
  2. splits are disjoint by game and match the configured seasons
  3. RE matrix and shrinkage mean (mu) are reproduced from TRAINING rows alone
  4. pitcher ratings use only games on earlier dates (independent recompute)
  5. usage features use only earlier days; bf_in_game only earlier events of this game
  6. the target is the runs from this event to the end of the half-inning
Only training and validation rows are read; test outcomes stay unread.
Writes reports/y1/leakage.json and exits non-zero if any check fails.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .model_y1 import POST_EVENT, RAW_INPUTS


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=3000, help="sampled rows per recompute check")
    ap.add_argument("--out", default="reports/y1/leakage.json")
    args = ap.parse_args(argv)
    cfg = Config.from_json(args.config)
    out = Path(cfg.out_dir)
    rng = np.random.default_rng(7)

    s = pd.read_parquet(out / "states.parquet", filters=[("split", "in", ["train", "validation"])])
    games = pd.read_parquet(out / "games.parquet")
    params = json.loads((out / "rating_params.json").read_text())
    re_saved = pd.read_csv(out / "re_matrix.csv", index_col=0)["expected_runs"]
    checks = {}

    def record(name, ok, **detail):
        checks[name] = {"pass": bool(ok), **detail}
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


    # 1. features ------------------------------------------------------------------
    record("no_outcome_columns_in_features", not (POST_EVENT & RAW_INPUTS),
           overlap=sorted(POST_EVENT & RAW_INPUTS))

    # 2. splits --------------------------------------------------------------------
    per_game = games.groupby("gid")["split"].nunique()
    expected = games["season"].map(cfg.split_of)
    record("splits_disjoint_by_game", per_game.max() == 1 and (expected == games["split"]).all(),
           games_by_split=games["split"].value_counts().to_dict())

    # 3. train-only parameters -----------------------------------------------------
    tr = s[(s["split"] == "train") & (s["inning"] <= 8) & (s["half_truncated"] == 0)]
    re_train = tr.groupby("state24")["runs_rest_half"].mean()
    re_trval = (s[(s["inning"] <= 8) & (s["half_truncated"] == 0)]
                .groupby("state24")["runs_rest_half"].mean())
    d_tr = float((re_train - re_saved).abs().max())
    d_trval = float((re_trval - re_saved).abs().max())
    record("re_matrix_fit_on_train_only", d_tr < 1e-9 and d_trval > 1e-6,
           max_diff_vs_train=d_tr, max_diff_if_validation_included=d_trval)

    rel = s[(s["split"] == "train") & (s["pitcher_is_starter"] == 0) & s["re24"].notna()]
    mu = rel["re24"].sum() / rel["is_pa"].sum()
    record("rating_mu_fit_on_train_only", abs(mu - params["mu"]) < 1e-12, mu_recomputed=mu, mu_saved=params["mu"])

    # 4. ratings use only earlier dates ----------------------------------------------
    q = (s[(s["pitcher_is_starter"] == 0) & (s["season"] >= 2018)]
         .drop_duplicates(["pitcher", "date"]).sample(args.n, random_state=1))
    daily = (s[s["re24"].notna()].groupby(["pitcher", "season", "date"])
             .agg(re24=("re24", "sum"), bf=("is_pa", "sum")).reset_index())
    m = q[["pitcher", "season", "date", "pitcher_rating"]].reset_index().merge(
        daily, on="pitcher", suffixes=("", "_h"))
    w = np.select([(m["season_h"] == m["season"]) & (m["date_h"] < m["date"]),
                   m["season_h"] == m["season"] - 1, m["season_h"] == m["season"] - 2],
                  [1.0, cfg.prior_season_weights[0], cfg.prior_season_weights[1]], 0.0)
    m["S"], m["N"] = w * m["re24"], w * m["bf"]
    agg = m.groupby("index")[["S", "N"]].sum().reindex(q.index, fill_value=0.0)
    k = params["k"]
    recomputed = (agg["S"] + k * params["mu"]) / (agg["N"] + k)
    diff = (recomputed - q["pitcher_rating"]).abs()
    # Counter-check: including same-day events would change ratings, so the test has power.
    same_day = m[(m["season_h"] == m["season"]) & (m["date_h"] == m["date"])]
    moved = same_day.groupby("index").size().reindex(q.index, fill_value=0).gt(0).mean()
    record("ratings_use_only_earlier_dates", diff.max() < 1e-9, n=len(q), max_abs_diff=float(diff.max()),
           share_where_same_day_data_exists=float(moved))

    # 5. usage features --------------------------------------------------------------
    apps = (s.groupby(["gid", "pitcher"]).agg(date=("date", "first"), game_num=("game_num", "first"),
                                             bf=("is_pa", "sum")).reset_index())
    u = (s[(s["pitcher_is_starter"] == 0) & (s["season"] >= 2017)]
         .drop_duplicates(["gid", "pitcher"]).sample(args.n, random_state=2))
    mu_ = u[["gid", "pitcher", "date", "pitched_d1", "pitched_d2", "bf_d1", "apps_prev4"]].reset_index().merge(
        apps.rename(columns={"gid": "gid_h", "date": "date_h"}), on="pitcher")
    mu_["lag"] = (mu_["date"] - mu_["date_h"]).dt.days
    g = mu_.assign(d1=mu_["lag"] == 1, d2=mu_["lag"] == 2, p4=mu_["lag"].between(1, 4),
                   bf1=np.where(mu_["lag"] == 1, mu_["bf"], 0)).groupby("index")
    rec = pd.DataFrame({"pitched_d1": g["d1"].any().astype(int), "pitched_d2": g["d2"].any().astype(int),
                        "bf_d1": g["bf1"].sum(), "apps_prev4": g["p4"].sum()}).reindex(u.index)
    bad = {c: int((rec[c] != u[c]).sum()) for c in rec}
    record("usage_uses_only_earlier_days", sum(bad.values()) == 0, n=len(u), mismatches=bad)

    ss = s.sort_values("row_idx")
    earlier = ss.groupby(["gid", "pitcher"])["is_pa"].cumsum() - ss["is_pa"]
    first = ~ss.duplicated(["gid", "pitcher"])
    record("bf_in_game_counts_only_earlier_events",
           (earlier == ss["pitcher_bf_in_game"]).all() and (ss.loc[first, "pitcher_bf_in_game"] == 0).all(),
           mismatches=int((earlier != ss["pitcher_bf_in_game"]).sum()))

    # 6. target definition -----------------------------------------------------------
    t = s.sample(args.n, random_state=3)[["half_id", "row_idx", "scored_rest_half"]].reset_index()
    tm = t.merge(s[["half_id", "row_idx", "runs"]], on="half_id", suffixes=("", "_h"))
    runs_after = tm[tm["row_idx_h"] >= tm["row_idx"]].groupby("index")["runs"].sum().reindex(t["index"]).fillna(0)
    record("target_is_runs_from_this_event_to_end_of_half",
           ((runs_after.values > 0).astype(int) == t["scored_rest_half"].values).all(), n=len(t))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(checks, indent=2, default=float))
    return 0 if all(c["pass"] for c in checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
