"""Unit tests for the decision simulator with hand-built run models (no data needed).

Run with `pytest -q tests/test_sim.py` or `python tests/test_sim.py`.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bullpen.runs_model import RunModel  # noqa: E402
from bullpen.simulator import SimSettings, simulate  # noqa: E402

PROFILE = dict(runs_saved_per100=0.0, rating_eff_bf=100.0, pitched_d1=0.0, pitched_d2=0.0,
               bf_d1=0.0, apps_prev4=0.0, bf_in_game=0.0)
PITCHER_FEATS = ["pitcher_runs_saved_per100", "log_rating_eff_bf", "pitched_d1", "pitched_d2",
                 "bf_d1", "apps_prev4", "pitcher_bf_in_game"]


def stub_model(class_logits=(0, 0, 0, 0), quality_beta=0.0, with_pitcher=True):
    b0 = np.array(class_logits, float)
    beta = {"fld_is_home": np.zeros(4), "extra_runner_era": np.zeros(4)}
    if with_pitcher:
        beta.update({f: np.zeros(4) for f in PITCHER_FEATS})
        beta["pitcher_runs_saved_per100"] = np.array([0.0, -1, -2, -3]) * quality_beta
    return RunModel(name="stub", b0=b0, t_inning=np.zeros((10, 4)), t_state=np.zeros((24, 4)),
                    t_margin=np.zeros((11, 4)), beta=beta, tail_cdf=np.array([1.0]))


def make_case(inning=9, margin=1, t_home=1, outs=0, bases="000", ratings=(3.0, 0.0, -1.0), actual=1):
    dec = pd.DataFrame([dict(decision_id="g_T_2", entry_inning=inning, entry_margin=margin, fld_is_home=t_home,
                             entry_outs=outs, entry_bases=bases, season=2023)])
    cand = pd.DataFrame([dict(decision_id="g_T_2", pitcher=f"p{i}", available=1, avail_rank=i + 1,
                              is_actual=int(i == actual), **{**PROFILE, "runs_saved_per100": r})
                         for i, r in enumerate(ratings)])
    return dec, cand


OPP = {"all": PROFILE, **{str(k): PROFILE for k in range(-3, 4)}}


def wp(model, dec, cand, n=4000, strategies=("A", "B")):
    return simulate(model, dec, cand, strategies, SimSettings(n_sims=n, opp_profile=OPP))["wp"]


def test_no_runs_means_leader_wins():
    never = stub_model((50, 0, 0, 0))
    for margin, expect in ((1, 1.0), (-1, 0.0), (0, 0.5)):
        for home in (0, 1):
            dec, cand = make_case(inning=8, margin=margin, t_home=home)
            r = wp(never, dec, cand, n=50)
            assert np.allclose(r["A"], expect) and np.allclose(r["B"], expect), (margin, home, r)


def test_walk_off_and_home_skips_bottom():
    one = stub_model((0, 50, 0, 0))          # every half-inning scores exactly one run
    # T home, leads by 1, top 9: opponent ties it, T walks off in the bottom 9th.
    dec, cand = make_case(inning=9, margin=1, t_home=1)
    assert np.allclose(wp(one, dec, cand, n=20)["A"], 1.0)
    # T away fielding the bottom 9th while tied: opponent's run is a walk-off loss.
    dec, cand = make_case(inning=9, margin=0, t_home=0)
    assert np.allclose(wp(one, dec, cand, n=20)["A"], 0.0)
    # T away leads by 2 entering bottom 9th: opponent scores one, T wins.
    dec, cand = make_case(inning=9, margin=2, t_home=0)
    assert np.allclose(wp(one, dec, cand, n=20)["A"], 1.0)


def test_placebo_without_pitcher_features_is_exactly_zero():
    m = stub_model((1.0, 0, -0.5, -1.0), with_pitcher=False)
    dec, cand = make_case(inning=7, margin=1, t_home=0)
    r = wp(m, dec, cand, n=2000)
    assert r["A"][0] == r["B"][0]


def test_identical_pitchers_identical_results_crn():
    m = stub_model((1.0, 0, -0.5, -1.0), quality_beta=0.5)
    dec, cand = make_case(inning=8, margin=0, ratings=(1.0, 1.0, 1.0))
    r = wp(m, dec, cand, n=2000)
    assert r["A"][0] == r["B"][0]


def test_best_now_helps_in_ninth_when_save_never_comes():
    # 9th inning, T home leading by 1: if he is saved there is no later high-leverage half.
    m = stub_model((1.0, 0, -0.5, -1.0), quality_beta=0.4)
    dec, cand = make_case(inning=9, margin=1, t_home=1)
    r = wp(m, dec, cand, n=20000)
    assert r["A"][0] > r["B"][0]


def test_reserve_rule_symmetry_seventh_inning():
    # 7th, T leading by 2, nothing can happen to the margin except via pitchers' runs. With a
    # very good reserve available for the 8th either way, A and B only swap who pitches 7th/8th:
    # their WPs should be close (difference well under the 9th-inning case).
    m = stub_model((1.0, 0, -0.5, -1.0), quality_beta=0.4)
    dec7, cand7 = make_case(inning=7, margin=2, t_home=1)
    dec9, cand9 = make_case(inning=9, margin=1, t_home=1)
    d7 = np.diff([wp(m, dec7, cand7, n=20000)[k][0] for k in ("B", "A")])[0]
    d9 = np.diff([wp(m, dec9, cand9, n=20000)[k][0] for k in ("B", "A")])[0]
    assert abs(d7) < d9


def test_actual_strategy_uses_actual_pitcher():
    m = stub_model((1.0, 0, -0.5, -1.0), quality_beta=0.4)
    dec, cand = make_case(inning=9, margin=1, t_home=1, actual=1)     # actual = the rank-2 reliever
    r = wp(m, dec, cand, n=3000, strategies=("B", "actual"))
    assert r["B"][0] == r["actual"][0]


def test_test_split_requires_confirmation():
    from bullpen.evaluate_sim import main
    try:
        main(["--split", "test", "--out", "/nonexistent_dir_for_test"])
    except SystemExit as e:
        assert "--confirm-test-once" in str(e)
    else:
        raise AssertionError("test split ran without confirmation")


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1; print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if fails else 0)
