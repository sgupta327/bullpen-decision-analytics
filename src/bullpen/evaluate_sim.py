"""Decision simulation: Strategy A (best reliever now) vs Strategy B (save him), SPEC §4.2-4.3.

    python -m bullpen.evaluate_sim --config configs/default.json                 # validation (2023)
    python -m bullpen.evaluate_sim --config configs/default.json --split test --confirm-test-once

Validation run (safe to repeat):
  1. Y1b run-distribution ladder M0/M1/M2 fit on training, scored on 2023.
  2. Simulator self-consistency: simulate each team's ACTUAL choice at 2023 decision points
     and check calibration against realised wins (vs an empirical win-expectancy table).
     Run with and without the 2023 intercept recalibration; the choice is frozen for test.
  3. dWP = P(win | A) - P(win | B) per decision point; mean and pre-specified subgroups
     with game-clustered bootstrap intervals; Monte Carlo error.
  4. Model-parameter uncertainty: refit the run model on game-bootstrap resamples of the
     training seasons and re-simulate; combined interval = parameter x sampling.
  5. Assumption tests: reserve rule, availability rule (strict/lenient builds), recalibration,
     training without 2020, opponent bullpen quality, and a placebo (no pitcher features -> dWP = 0).
Writes reports/sim/{report.md, metrics.json, decisions_<split>.csv, *.png, frozen_settings.json}.

Test run: refuses unless frozen_settings.json exists (written by the validation run) and
the test set has not been opened before; it records the opening in test_opened.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .model_y1 import evaluate as evaluate_binary
from .runs_model import RunModel, empirical_m0, evaluate_ladder, load_relief_states, read_table
from .simulator import RESERVE_RULES, SimSettings, league_average_profile, simulate

SEED = 20261022
SUBGROUPS = ["inning", "margin", "bases", "gap_tercile", "home"]
SUBGROUP_ORDER = {
    "inning": ["7", "8", "9"],
    "margin": ["trailing 1-2", "tied", "leading 1", "leading 2-3"],
    "bases": ["bases empty", "runners on"],
    "gap_tercile": ["small gap", "middle gap", "large gap"],
    "home": ["home", "away"],
}


# ----------------------------------------------------------------- data -------
def load_decisions(cfg: Config, split: str):
    m = json.loads((Path(cfg.out_dir) / "manifest.json").read_text())
    seasons = {"validation": cfg.val_seasons, "test": cfg.test_seasons, "train": cfg.train_seasons}[split]
    dec = read_table(m["outputs"]["decisions"], splits=[split], str_cols=("entry_bases",))
    dec["entry_bases"] = dec["entry_bases"].str.zfill(3)
    cand = read_table(m["outputs"]["candidates"], seasons=list(seasons))
    cand = cand[cand["decision_id"].isin(dec["decision_id"])].reset_index(drop=True)
    return dec.sort_values("decision_id").reset_index(drop=True), cand


def subgroup_labels(dec: pd.DataFrame, gap_cuts) -> pd.DataFrame:
    mg = dec["entry_margin"]
    return pd.DataFrame({
        "inning": dec["entry_inning"].astype(str),
        "margin": np.select([mg < 0, mg == 0, mg == 1], ["trailing 1-2", "tied", "leading 1"], "leading 2-3"),
        "bases": np.where(dec["entry_bases"] == "000", "bases empty", "runners on"),
        "gap_tercile": pd.cut(dec["rating_gap_per100"], [-np.inf, *gap_cuts, np.inf],
                              labels=SUBGROUP_ORDER["gap_tercile"]).astype(str),
        "home": np.where(dec["fld_is_home"] == 1, "home", "away"),
    })


# ------------------------------------------------------------ statistics ------
class GameBootstrap:
    """Game-clustered bootstrap weights shared by every estimate in one run."""

    def __init__(self, gid, B: int, seed: int):
        self.codes, uniq = pd.factorize(np.asarray(gid))
        self.G, self.B = len(uniq), B
        rng = np.random.default_rng(seed)
        self.W = np.array([np.bincount(rng.integers(0, self.G, self.G), minlength=self.G) for _ in range(B)],
                          dtype=float)

    def means(self, x, mask=None):
        """Bootstrap distribution of the mean of x over the rows in mask."""
        mask = np.ones(len(x), bool) if mask is None else np.asarray(mask)
        s = np.bincount(self.codes[mask], np.asarray(x, float)[mask], self.G)
        n = np.bincount(self.codes[mask], minlength=self.G).astype(float)
        return (self.W @ s) / (self.W @ n)


def summarize(d, se_var, boot: GameBootstrap, n_sims: int, mask=None) -> dict:
    mask = np.ones(len(d), bool) if mask is None else np.asarray(mask)
    bm = boot.means(d, mask)
    k = mask.sum()
    return {"n": int(k), "n_games": int(len(np.unique(boot.codes[mask]))),
            "mean": float(d[mask].mean()), "ci": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
            "boot_se": float(bm.std()), "mc_se": float(np.sqrt(se_var[mask].sum() / n_sims) / k),
            "share_positive": float((d[mask] > 0).mean()), "p_boot_le_0": float((bm <= 0).mean())}


def subgroup_table(d, se_var, labels: pd.DataFrame, boot, n_sims) -> dict:
    out = {}
    for g in SUBGROUPS:
        out[g] = {lv: summarize(d, se_var, boot, n_sims, (labels[g] == lv).to_numpy())
                  for lv in SUBGROUP_ORDER[g] if (labels[g] == lv).any()}
    return out


def we0_table(train_dec: pd.DataFrame):
    """Empirical win expectancy at the decision state: inning x home x margin (training)."""
    key = ["entry_inning", "fld_is_home", "entry_margin"]
    t = train_dec.groupby(key)["fld_win"].agg(["sum", "count"])
    return ((t["sum"] + 1) / (t["count"] + 2)).rename("we0"), key


# ------------------------------------------------------------- pipeline -------
def fit_run_model(relief: pd.DataFrame, recal: bool, name="M2", weights=None, exclude_seasons=()):
    tr = relief[(relief["split"] == "train") & ~relief["season"].isin(exclude_seasons)]
    w = None if weights is None else weights[tr.index.to_numpy()]
    model = RunModel.fit(tr, name, weights=w)
    if recal:
        model.fit_recalibration(relief[relief["split"] == "validation"])
    return model


def run_sim(model, dec, cand, settings: SimSettings, strategies=("A", "B")):
    r = simulate(model, dec, cand, strategies, settings)
    return r["wp"], r["diff_var"]["A-B"]


def analyse(cfg: Config, split: str, out_dir: Path, opts) -> dict:
    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:7.0f}s] {msg}", flush=True)
    rng_seed = opts.seed
    relief = load_relief_states(cfg)                       # train + validation only, never test
    train_dec = read_table(json.loads((Path(cfg.out_dir) / "manifest.json").read_text())["outputs"]["decisions"],
                           splits=["train"], str_cols=("entry_bases",))
    gap_cuts = np.quantile(train_dec["rating_gap_per100"].dropna(), [1 / 3, 2 / 3]).tolist()
    profile = league_average_profile(relief[relief["split"] == "train"])
    dec, cand = load_decisions(cfg, split)
    log(f"{split}: {len(dec):,} decision points in {dec['gid'].nunique():,} games")
    labels = subgroup_labels(dec, gap_cuts)
    boot = GameBootstrap(dec["gid"], opts.boot, rng_seed)
    y = dec["fld_win"].to_numpy(float)
    res = {"meta": {"split": split, "n_decisions": len(dec), "n_games": int(dec["gid"].nunique()),
                    "gap_cuts_from_train": gap_cuts, "opp_profile": profile, "boot": opts.boot}}

    # 1. Y1b ladder (validation only; it is a model-selection step)
    if split == "validation":
        tr, va = relief[relief["split"] == "train"], relief[relief["split"] == "validation"]
        models = {n: RunModel.fit(tr, n) for n in ("M1", "M2")}
        m0 = empirical_m0(tr)
        preds = {"M0": m0.loc[va["state24"]].to_numpy(), "M1": models["M1"].predict_proba(va),
                 "M2": models["M2"].predict_proba(va)}
        res["y1b_ladder"] = evaluate_ladder(va, preds, opts.boot, np.random.default_rng(rng_seed))
        rec = RunModel.fit(tr, "M2")
        rec.fit_recalibration(va)
        res["y1b_ladder"]["M2_recal_offsets"] = rec.recal.tolist()
        res["y1b_ladder"]["M2_recal_pred_share"] = rec.predict_proba(va).mean(0).round(5).tolist()
        log("Y1b ladder done")

    # 2-3. main simulations: both recalibration options, all three strategies
    we0, key = we0_table(train_dec)
    p_we0 = dec[key].merge(we0, left_on=key, right_index=True, how="left")["we0"].fillna(
        train_dec["fld_win"].mean()).to_numpy()
    sims, cal = {}, {}
    recal_options = [opts.recal] if split == "test" else [True, False]
    for rc in recal_options:
        model = fit_run_model(relief, recal=rc)
        st = SimSettings(n_sims=opts.n_sims, seed=rng_seed, opp_profile=profile)
        r = simulate(model, dec, cand, ("A", "B", "actual"), st)
        tag = "recal" if rc else "no_recal"
        sims[tag] = r
        cal[tag] = evaluate_binary(y, {"WE0_table": p_we0, "SIM_actual": r["wp"]["actual"]},
                                   dec["gid"].to_numpy(), opts.boot, np.random.default_rng(rng_seed))
        res.setdefault("model", {})[tag] = model.to_dict()
        log(f"simulated A/B/actual ({tag})")
    res["self_consistency"] = cal
    primary = "recal" if opts.recal else "no_recal"
    res["meta"]["primary"] = primary
    r = sims[primary]
    wpA, wpB = r["wp"]["A"], r["wp"]["B"]
    d, dv = wpA - wpB, r["diff_var"]["A-B"]
    res["dwp"] = {"overall": summarize(d, dv, boot, opts.n_sims),
                  "wp_A": float(wpA.mean()), "wp_B": float(wpB.mean()), "wp_actual": float(r["wp"]["actual"].mean()),
                  "observed_win": float(y.mean()),
                  "mc_se_per_decision_median": float(np.median(np.sqrt(dv / opts.n_sims))),
                  "mc_se_per_decision_max": float(np.max(np.sqrt(dv / opts.n_sims))),
                  "subgroups": subgroup_table(d, dv, labels, boot, opts.n_sims)}
    grid = pd.DataFrame({"inning": labels["inning"], "margin": labels["margin"], "d": d})
    res["dwp"]["inning_x_margin"] = (grid.groupby(["inning", "margin"])["d"].agg(["mean", "count"])
                                     .reset_index().to_dict("records"))
    other = "no_recal" if primary == "recal" else "recal"
    per = dec[["decision_id", "gid", "season", "entry_inning", "entry_outs", "entry_bases", "entry_margin",
               "fld_is_home", "n_available", "best_pitcher", "alt_pitcher", "best_runs_saved_per100",
               "alt_runs_saved_per100", "rating_gap_per100", "actual_choice", "fld_win"]].copy()
    per["wp_A"], per["wp_B"], per["wp_actual"] = wpA, wpB, r["wp"]["actual"]
    per["dwp"], per["mc_se"] = d, np.sqrt(dv / opts.n_sims)

    # descriptive, confounded: realised win rate by the team's actual choice
    res["descriptive_actual_choice"] = (dec.groupby("actual_choice", observed=True)["fld_win"]
                                        .agg(["mean", "count"]).round(4).reset_index().to_dict("records"))

    # 4. parameter uncertainty: refit on game-bootstrap resamples of the training seasons
    tr_rows = relief["split"] == "train"
    codes, uniq = pd.factorize(relief.loc[tr_rows, "gid"])
    prng = np.random.default_rng(rng_seed + 1)
    refit_means, refit_d = [], []
    st_refit = SimSettings(n_sims=opts.n_sims_refit, seed=rng_seed, opp_profile=profile)
    for b in range(opts.refits):
        gw = np.bincount(prng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)).astype(float)
        w = np.zeros(len(relief))
        w[np.flatnonzero(tr_rows)] = gw[codes]
        model_b = fit_run_model(relief, recal=opts.recal, weights=w)
        wp_b, _ = run_sim(model_b, dec, cand, st_refit)
        db = wp_b["A"] - wp_b["B"]
        refit_d.append(db)
        refit_means.append(float(db.mean()))
        if (b + 1) % 5 == 0:
            log(f"parameter refit {b + 1}/{opts.refits}")
    refit_d = np.array(refit_d)
    comb = np.array([boot.W[i] @ np.bincount(boot.codes, refit_d[i % opts.refits], boot.G)
                     / (boot.W[i] @ np.bincount(boot.codes, minlength=boot.G)) for i in range(opts.boot)])
    res["dwp"]["parameter_uncertainty"] = {
        "refits": opts.refits, "n_sims_per_refit": opts.n_sims_refit,
        "refit_means": refit_means, "param_sd": float(np.std(refit_means, ddof=1)),
        "param_interval": [float(np.percentile(refit_means, 2.5)), float(np.percentile(refit_means, 97.5))],
        "combined_ci": [float(np.percentile(comb, 2.5)), float(np.percentile(comb, 97.5))],
        "combined_p_le_0": float((comb <= 0).mean()),
    }
    for g in SUBGROUPS:
        for lv, row in res["dwp"]["subgroups"][g].items():
            msk = (labels[g] == lv).to_numpy()
            cg = np.array([boot.W[i] @ np.bincount(boot.codes[msk], refit_d[i % opts.refits][msk], boot.G)
                           / (boot.W[i] @ np.bincount(boot.codes[msk], minlength=boot.G)) for i in range(opts.boot)])
            row["combined_ci"] = [float(np.percentile(cg, 2.5)), float(np.percentile(cg, 97.5))]
    per["dwp_refit_sd"] = refit_d.std(0, ddof=1)
    log("parameter uncertainty done")

    # 5. assumption tests
    sens = {}
    base_model = fit_run_model(relief, recal=opts.recal)
    st_s = SimSettings(n_sims=opts.n_sims_sens, seed=rng_seed, opp_profile=profile)

    def add(name, wp, dvar, n_sims, desc, dec_=dec, boot_=boot):
        dd = wp["A"] - wp["B"]
        sens[name] = {"description": desc, **summarize(dd, dvar, boot_, n_sims)}

    if split == "validation":
        add(f"recalibration: {other}", sims[other]["wp"], sims[other]["diff_var"]["A-B"], opts.n_sims,
            "the other recalibration option")
    for rule in RESERVE_RULES:
        if rule == "default":
            continue
        wp, dvar = run_sim(base_model, dec, cand, SimSettings(**{**asdict(st_s), "reserve_rule": rule}))
        lo = RESERVE_RULES[rule]
        add(f"reserve rule: {rule}", wp, dvar, opts.n_sims_sens,
            f"held reliever enters first later half-inning with inning >= {lo[0]} and T margin {lo[1]}..{lo[2]}")
    m_no2020 = fit_run_model(relief, recal=opts.recal, exclude_seasons=(2020,))
    wp, dvar = run_sim(m_no2020, dec, cand, st_s)
    add("training without 2020", wp, dvar, opts.n_sims_sens, "run model refit without the 60-game season")
    strong = {k: dict(v, runs_saved_per100=v["runs_saved_per100"] + 1.0) for k, v in profile.items()}
    wp, dvar = run_sim(base_model, dec, cand, SimSettings(**{**asdict(st_s), "opp_profile": strong}))
    add("strong opponent bullpen", wp, dvar, opts.n_sims_sens,
        "every opponent reliever rated 1 run saved per 100 BF better than the training average")
    m1 = fit_run_model(relief, recal=opts.recal, name="M1")
    wp, dvar = run_sim(m1, dec, cand, SimSettings(**{**asdict(st_s), "n_sims": 200}))
    add("placebo: no pitcher features (M1)", wp, dvar, 200, "must be exactly 0: strategies differ only by pitcher")
    for tag, path in (("strict", opts.strict_config), ("lenient", opts.lenient_config)):
        c2 = Config.from_json(path)
        if not (Path(c2.out_dir) / "manifest.json").exists():
            sens[f"availability: {tag}"] = {"description": f"skipped: run `make sensitivity` to build {c2.out_dir}"}
            continue
        d2, cand2 = load_decisions(c2, split)
        wp, dvar = run_sim(base_model, d2, cand2, st_s)
        add(f"availability: {tag}", wp, dvar, opts.n_sims_sens, f"decisions rebuilt with {path}", d2,
            GameBootstrap(d2["gid"], opts.boot, rng_seed))
    res["sensitivity"] = sens
    log("assumption tests done")

    res["meta"]["seconds"] = round(time.time() - t0, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    per.to_csv(out_dir / f"decisions_{split}.csv", index=False)
    (out_dir / f"metrics_{split}.json").write_text(json.dumps(res, indent=1, default=float))
    return res


# ------------------------------------------------------------- CLI ------------
def spec_hash(path="docs/SPEC.md") -> str | None:
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--out", default="reports/sim")
    ap.add_argument("--n-sims", type=int, default=2000)
    ap.add_argument("--n-sims-refit", type=int, default=400)
    ap.add_argument("--n-sims-sens", type=int, default=1000)
    ap.add_argument("--refits", type=int, default=40)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--recal", choices=["yes", "no"], default=None,
                    help="validation: primary option to report (default yes); test: taken from frozen settings")
    ap.add_argument("--strict-config", default="configs/sensitivity_strict.json")
    ap.add_argument("--lenient-config", default="configs/sensitivity_lenient.json")
    ap.add_argument("--confirm-test-once", action="store_true")
    args = ap.parse_args(argv)
    cfg = Config.from_json(args.config)
    out = Path(args.out)
    frozen_path, lock = out / "frozen_settings.json", out / "test_opened.json"

    if args.split == "test":
        if not args.confirm_test_once:
            raise SystemExit("The test set is evaluated once (SPEC §2.4). Re-run with --confirm-test-once.")
        if lock.exists():
            raise SystemExit(f"Test set already opened ({lock}); results are in {out}/metrics_test.json.")
        if not frozen_path.exists():
            raise SystemExit("Run the validation analysis first; it writes frozen_settings.json.")
        frozen = json.loads(frozen_path.read_text())
        for k in ("n_sims", "n_sims_refit", "n_sims_sens", "refits", "boot", "seed"):
            setattr(args, k, frozen[k])
        args.recal = frozen["recal"]
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(json.dumps({"opened_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                    "spec_sha256": spec_hash(), "frozen_settings": frozen}, indent=2))
    else:
        args.recal = args.recal != "no"

    res = analyse(cfg, args.split, out, args)
    if args.split == "validation":
        frozen = {k: getattr(args, k) for k in ("n_sims", "n_sims_refit", "n_sims_sens", "refits", "boot", "seed",
                                                "recal")}
        frozen.update({"reserve_rule": "default", "gap_cuts_from_train": res["meta"]["gap_cuts_from_train"],
                       "spec_sha256_at_freeze": spec_hash(),
                       "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        frozen_path.write_text(json.dumps(frozen, indent=2))
    from .sim_report import write_report
    write_report(out, args.split)
    o = res["dwp"]["overall"]
    print(json.dumps({"dWP_mean": o["mean"], "ci": o["ci"], "mc_se": o["mc_se"],
                      "combined_ci": res["dwp"]["parameter_uncertainty"]["combined_ci"]}, indent=1))


if __name__ == "__main__":
    main()
