"""Game-state logic on a hand-built game: scores, margins, targets, walk-offs, appearances, rest."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bullpen.availability import UsageIndex  # noqa: E402
from bullpen.config import Config  # noqa: E402
from bullpen.states import build_appearances, build_states  # noqa: E402

GID = "HOM202306010"


def ev(inning, bat_home, outs, runs=0, br=(0, 0, 0)):
    pitcher = {(1, 0): "h_start", (9, 0): "h_relief", (1, 1): "v_start", (9, 1): "v_relief"}[(inning, bat_home)]
    return dict(inning=inning, bat_home=bat_home, outs_pre=outs, runs=runs, pitcher=pitcher,
                br1=br[0], br2=br[1], br3=br[2])


def toy_game() -> pd.DataFrame:
    """Visitors score 1 in the 1st; home scores 2 in the bottom 9th for a walk-off 2-1 win."""
    rows = [ev(1, 0, 0), ev(1, 0, 0, runs=1), ev(1, 0, 1), ev(1, 0, 2),       # top 1: VIS 1
            ev(1, 1, 0), ev(1, 1, 1), ev(1, 1, 2),                            # bottom 1
            ev(9, 0, 0), ev(9, 0, 1), ev(9, 0, 2),                            # top 9
            ev(9, 1, 0), ev(9, 1, 0, br=(1, 0, 0)), ev(9, 1, 0, runs=2, br=(1, 1, 0))]  # walk-off
    df = pd.DataFrame(rows)
    df["gid"], df["season"], df["date"], df["game_num"] = GID, 2023, pd.Timestamp("2023-06-01"), 0
    df["home_team"] = "HOM"
    df["bat_team"] = df["bat_home"].map({0: "VIS", 1: "HOM"})
    df["fld_team"] = df["bat_home"].map({0: "HOM", 1: "VIS"})
    df["row_order"] = range(len(df))
    df["is_pa"] = 1
    return df


def built():
    return build_states(toy_game(), Config())


def test_scores_and_margin_before_each_event():
    s, _ = built()
    assert s["vis_score_pre"].tolist()[:4] == [0, 0, 1, 1]          # the run counts after its event
    b9 = s[(s["inning"] == 9) & (s["bat_home"] == 1)]
    assert (b9["fld_margin"] == 1).all()                               # visitors lead 1-0 in the field
    t9 = s[(s["inning"] == 9) & (s["bat_home"] == 0)]
    assert (t9["fld_margin"] == -1).all() and (t9["fld_is_home"] == 1).all()


def test_runs_rest_of_half_inning_target():
    s, _ = built()
    t1 = s[(s["inning"] == 1) & (s["bat_home"] == 0)]
    assert t1["runs_rest_half"].tolist() == [1, 1, 0, 0]               # includes this event's runs
    assert t1["scored_rest_half"].tolist() == [1, 1, 0, 0]
    b9 = s[(s["inning"] == 9) & (s["bat_home"] == 1)]
    assert b9["runs_rest_half"].tolist() == [2, 2, 2]


def test_walk_off_half_is_flagged_and_win_targets():
    s, games = built()
    trunc = s.groupby(["inning", "bat_home"])["half_truncated"].max()
    assert trunc[(9, 1)] == 1 and trunc.drop((9, 1)).eq(0).all()
    g = games.iloc[0]
    assert (g["home_final"], g["vis_final"], g["home_win"]) == (2, 1, 1)
    assert g["home_team"] == "HOM" and g["vis_team"] == "VIS" and g["sched_innings"] == 9
    assert (s.loc[s["bat_home"] == 0, "fld_win"] == 1).all()           # home team fields in the top
    assert (s.loc[s["bat_home"] == 1, "fld_win"] == 0).all()


def test_base_out_state_key():
    s, _ = built()
    assert s["state24"].iloc[-1] == "0_110" and s["state24"].iloc[0] == "0_000"


def test_appearances_starter_flag_and_batters_faced():
    s, _ = built()
    apps = build_appearances(s).set_index("pitcher")
    assert apps.loc["h_start", "is_starter"] == 1 and apps.loc["h_relief", "is_starter"] == 0
    assert apps.loc["h_relief", "app_order"] == 2
    assert apps.loc["h_start", "bf"] == 4 and apps.loc["v_relief", "bf"] == 3
    assert apps.loc["h_relief", "entry_inning"] == 9 and apps.loc["h_relief", "entry_margin"] == -1


def test_usage_ignores_this_game_and_later_games():
    d = pd.Timestamp("2023-06-10")
    apps = pd.DataFrame([
        dict(pitcher="p", date=d - pd.Timedelta(days=2), game_num=0, bf=4, is_starter=0, fld_team="T"),
        dict(pitcher="p", date=d - pd.Timedelta(days=1), game_num=0, bf=12, is_starter=0, fld_team="T"),
        dict(pitcher="p", date=d, game_num=1, bf=3, is_starter=0, fld_team="T"),   # doubleheader game 1
        dict(pitcher="p", date=d, game_num=2, bf=5, is_starter=0, fld_team="T"),   # game 2
    ])
    idx = UsageIndex(apps)
    g1 = idx.usage("p", d.toordinal(), 1, 4)
    g2 = idx.usage("p", d.toordinal(), 2, 4)
    assert g1["pitched_earlier_today"] == 0 and g2["pitched_earlier_today"] == 1
    assert (g1["pitched_d1"], g1["pitched_d2"], g1["bf_d1"], g1["apps_prev4"]) == (1, 1, 12, 2)
