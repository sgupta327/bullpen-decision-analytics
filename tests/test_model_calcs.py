"""Model calculations: feature encoding, recalibration, metrics and the shrunk reliever rating."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bullpen.config import Config  # noqa: E402
from bullpen.model_y1 import STATE24, evaluate, make_model, prepare, recalib  # noqa: E402
from bullpen.ratings import ratings_asof  # noqa: E402


def raw_states(n, rng):
    return pd.DataFrame({
        "inning": rng.integers(1, 13, n), "state24": rng.choice(STATE24, n),
        "fld_margin": rng.integers(-8, 9, n), "fld_is_home": rng.integers(0, 2, n),
        "extra_runner_era": rng.integers(0, 2, n), "pitcher_runs_saved_per100": rng.normal(0, 1, n),
        "pitcher_rating_eff_bf": rng.uniform(0, 800, n), "pitched_d1": rng.integers(0, 2, n),
        "pitched_d2": rng.integers(0, 2, n), "bf_d1": rng.integers(0, 40, n),
        "apps_prev4": rng.integers(0, 6, n), "pitcher_bf_in_game": rng.integers(0, 20, n),
    })


def test_prepare_buckets_and_caps():
    rng = np.random.default_rng(0)
    X = prepare(raw_states(2000, rng))
    assert set(X["inning_cat"]) <= {*map(str, range(1, 10)), "10+"} and "10+" in set(X["inning_cat"])
    assert X["margin_cat"].between(-5, 5).all()
    assert X["bf_d1"].max() <= 30 and X["apps_prev4"].max() <= 4 and X["pitcher_bf_in_game"].max() <= 12


def test_fixed_categories_survive_unseen_levels():
    rng = np.random.default_rng(1)
    tr = raw_states(3000, rng)
    tr["state24"] = "0_000"                                            # train sees one base-out state
    m = make_model("M1").fit(prepare(tr), rng.integers(0, 2, len(tr)))
    p = m.predict_proba(prepare(raw_states(500, rng)))[:, 1]           # scored on all 24 states
    assert np.isfinite(p).all()


def test_recalibration_recovers_intercept_and_slope():
    rng = np.random.default_rng(2)
    z = rng.normal(0, 1.2, 200_000)
    y = (rng.random(z.size) < expit(0.3 + 0.7 * z)).astype(float)
    a, b = recalib(z, y, np.ones_like(z), slope=True)
    assert abs(a - 0.3) < 0.02 and abs(b - 0.7) < 0.02


def test_metrics_on_known_predictions():
    rng = np.random.default_rng(3)
    n = 60_000
    p_true = expit(rng.normal(-1, 1, n))
    y = (rng.random(n) < p_true).astype(float)
    gid = np.repeat(np.arange(n // 30), 30)
    weak = expit(-1 + 0.2 * rng.normal(0, 1, n))                          # unrelated to the truth
    r = evaluate(y, {"weak": weak, "true": p_true}, gid, 200, np.random.default_rng(4))
    t = r["models"]["true"]
    ll = -np.mean(y * np.log(p_true) + (1 - y) * np.log(1 - p_true))
    assert abs(t["log_loss"] - ll) < 1e-12
    assert abs(t["cal_intercept"]) < 0.05 and abs(t["cal_slope"] - 1) < 0.05
    assert t["ece"] < 0.01
    d = r["differences"]["true-weak"]
    assert d["log_loss_diff"] < 0 and d["ci"][1] < 0
    lo, hi = t["log_loss_ci"]
    assert lo <= t["log_loss"] <= hi


def test_rating_shrinkage_formula_and_no_same_day_data():
    cfg = Config()
    params = {"mu": -0.01, "k": 400.0}
    daily = pd.DataFrame({
        "pitcher": ["a"] * 4, "season": [2021, 2022, 2023, 2023],
        "date": pd.to_datetime(["2021-07-01", "2022-07-01", "2023-05-01", "2023-06-01"]),
        "re24": [-3.0, -6.0, -2.0, 50.0], "bf": [100.0, 200.0, 50.0, 10.0],
    })
    q = pd.DataFrame({"pitcher": ["a", "new"], "season": [2023, 2023],
                      "date": pd.to_datetime(["2023-06-01", "2023-06-01"])})
    r = ratings_asof(q, daily, params, cfg).set_index("pitcher")
    S = -2.0 + 0.6 * -6.0 + 0.3 * -3.0                                 # the 2023-06-01 game is excluded
    N = 50.0 + 0.6 * 200.0 + 0.3 * 100.0
    assert np.isclose(r.loc["a", "rating"], (S + 400 * -0.01) / (N + 400))
    assert np.isclose(r.loc["a", "rating_eff_bf"], N)
    assert np.isclose(r.loc["new", "rating"], -0.01) and np.isclose(r.loc["new", "runs_saved_per100"], 0)
