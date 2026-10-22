"""Rest-of-game simulator for the bullpen decision (docs/SPEC.md §1.2, §3, §4.2).

Unit of simulation: the half-inning. From the decision point to the end of the game,
each half-inning's runs are drawn from the fitted Y1b model (0/1/2/3+, with 3+ expanded
from the empirical tail), given inning, starting base-out state, the fielding team's
margin, home/away, rules era and the pitcher on the mound.

Strategies for the deciding team T (pre-specified, SPEC §6):
  A  "best now":  rank-1 available reliever pitches now; rank-2 is the reserve.
  B  "save":      rank-2 pitches now; rank-1 is the reserve.
  actual          the pitcher who actually entered pitches now; the best other available
                  reliever is the reserve (used only for the self-consistency check).
Later half-innings in which T fields:
  * reserve rule: the reserve (if unused) pitches the first later half-inning that starts
    with inning >= 8 and T tied or leading by 1-3 (configurable for sensitivity runs);
  * otherwise the next unused "depth" reliever in rating order (rank 3, 4, ...) pitches;
    if the bullpen is exhausted, a league-average reliever pitches.
Each reliever pitches the rest of the half-inning he starts (one-inning relievers); this
also satisfies the three-batter minimum except in the rare case he faces fewer than three
batters and the half ends, which the half-inning model already allows.
The opponent's half-innings use the average training reliever profile for its current margin
(innings 7+, margin capped at +-3): leading teams use better relievers than trailing ones.

Game rules: walk-off ends the game in the bottom of the 9th or later; the home team does
not bat in the bottom of the 9th+ when leading; extra innings start with a runner on
second from 2020 on; games still tied after `max_inning` count as 0.5.

Common random numbers: every strategy uses the same uniform draws for the same
(decision, simulation, half-inning step), so A - B differences are paired and the Monte
Carlo noise largely cancels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model_y1 import STATE24
from .runs_model import RunModel

STATE_IDX = {s: i for i, s in enumerate(STATE24)}
S_EMPTY, S_GHOST = STATE_IDX["0_000"], STATE_IDX["0_010"]
PITCHER_COLS = ["runs_saved_per100", "rating_eff_bf", "pitched_d1", "pitched_d2", "bf_d1",
                "apps_prev4", "bf_in_game"]

RESERVE_RULES = {
    # name: (min inning, min T margin, max T margin) at the start of a later half-inning
    "default": (8, 0, 3),        # SPEC §1.2: inning 8+, tied or leading by 1-3
    "save_only": (9, 1, 3),      # sensitivity: classic save situations only
    "late_close": (8, -1, 1),    # sensitivity: inning 8+, within one run either way
}


@dataclass
class SimSettings:
    n_sims: int = 2000
    seed: int = 20261022
    reserve_rule: str = "default"
    max_inning: int = 20
    chunk: int = 150
    opp_profile: dict | None = None       # reliever profiles: {"all": {...}, "-3".."3": {...}}


OPP_MARGIN_CAP = 3


def _profile(df: pd.DataFrame) -> dict:
    return {"runs_saved_per100": float(df["pitcher_runs_saved_per100"].mean()),
            "rating_eff_bf": float(np.expm1(np.log1p(df["pitcher_rating_eff_bf"]).mean())),
            "pitched_d1": float(df["pitched_d1"].mean()), "pitched_d2": float(df["pitched_d2"].mean()),
            "bf_d1": float(df["bf_d1"].clip(0, 30).mean()),
            "apps_prev4": float(df["apps_prev4"].clip(0, 4).mean()), "bf_in_game": 0.0}


def league_average_profile(relief_train: pd.DataFrame) -> dict:
    """Average reliever features of training relief states in innings 7+, overall ("all", the
    bullpen-exhausted fallback) and by the fielding team's margin capped at +-3 (the opponent's
    pitcher: teams protecting a lead use better relievers than teams that trail)."""
    late = relief_train[relief_train["inning"] >= 7]
    mc = late["fld_margin"].clip(-OPP_MARGIN_CAP, OPP_MARGIN_CAP)
    out = {"all": _profile(late)}
    out.update({str(k): _profile(late[mc == k]) for k in range(-OPP_MARGIN_CAP, OPP_MARGIN_CAP + 1)})
    return out


# ------------------------------------------------------------ bullpens --------
def bullpen_orders(decisions: pd.DataFrame, candidates: pd.DataFrame, strategy: str) -> list[list[int]]:
    """For each decision, candidate row indices in slot order [now, reserve, depth...]."""
    cand = candidates.reset_index(drop=True)
    av = cand[cand["available"] == 1].sort_values(["decision_id", "avail_rank"])
    av_by = av.groupby("decision_id", sort=False).groups
    act_by = cand.index[cand["is_actual"] == 1].to_series().groupby(cand.loc[cand["is_actual"] == 1, "decision_id"]).first()
    orders = []
    for dec_id in decisions["decision_id"]:
        ranked = list(av_by.get(dec_id, []))
        if len(ranked) < 2:
            raise ValueError(f"{dec_id}: fewer than two available relievers")
        if strategy == "A":
            order = ranked
        elif strategy == "B":
            order = [ranked[1], ranked[0]] + ranked[2:]
        elif strategy == "actual":
            a = act_by[dec_id]
            rest = [i for i in ranked if i != a]
            order = [a] + rest
        else:
            raise ValueError(strategy)
        orders.append(order)
    return orders


def slot_logits(model: RunModel, candidates: pd.DataFrame, orders, profile: dict, n_slots: int):
    """(D, n_slots + 1, 4) pitcher logit per slot; the last slot is the league-average fallback."""
    cand = candidates.reset_index(drop=True).copy()
    cand["bf_in_game"] = 0.0
    pl = model.pitcher_logit(cand[PITCHER_COLS])
    avg = model.pitcher_logit(pd.DataFrame([profile["all"]])[PITCHER_COLS])[0]
    D = len(orders)
    out = np.broadcast_to(avg, (D, n_slots + 1, 4)).copy()
    n_have = np.zeros(D, dtype=int)
    for d, order in enumerate(orders):
        k = min(len(order), n_slots)
        out[d, :k] = pl[order[:k]]
        n_have[d] = k
    return out, n_have


# ------------------------------------------------------------- engine ---------
def simulate(model: RunModel, decisions: pd.DataFrame, candidates: pd.DataFrame,
             strategies=("A", "B"), settings: SimSettings = SimSettings(),
             keep_pairs=(("A", "B"),)) -> dict:
    """Simulate every decision under each strategy.

    Returns {"wp": {strategy: (D,)}, "diff_var": {"A-B": (D,)}} where diff_var is the
    per-simulation variance of the paired win difference (for Monte Carlo error).
    """
    dec = decisions.reset_index(drop=True)
    D = len(dec)
    profile = settings.opp_profile
    n_slots = int(candidates.groupby("decision_id")["available"].sum().max()) + 1
    slots = {}
    for s in strategies:
        orders = bullpen_orders(dec, candidates, s)
        slots[s] = slot_logits(model, candidates, orders, profile, n_slots)
    opp_logit = model.pitcher_logit(pd.DataFrame([profile[str(k)] for k in
                                                   range(-OPP_MARGIN_CAP, OPP_MARGIN_CAP + 1)])[PITCHER_COLS])
    rule = RESERVE_RULES[settings.reserve_rule]
    keep_pairs = [(a, b) for a, b in keep_pairs if a in strategies and b in strategies]

    wp = {s: np.empty(D) for s in strategies}
    dvar = {f"{a}-{b}": np.empty(D) for a, b in keep_pairs}
    entry_state = np.array([STATE_IDX[f"{o}_{str(b).zfill(3)}"]
                            for o, b in zip(dec["entry_outs"], dec["entry_bases"])])
    for c0 in range(0, D, settings.chunk):
        sl = slice(c0, min(c0 + settings.chunk, D))
        wins = {}
        for s in strategies:
            rng = np.random.default_rng([settings.seed, c0])
            wins[s] = _simulate_chunk(model, dec.iloc[sl], entry_state[sl], slots[s][0][sl],
                                      slots[s][1][sl], opp_logit, rule, settings, rng)
            wp[s][sl] = wins[s].mean(1)
        for a, b in keep_pairs:
            dvar[f"{a}-{b}"][sl] = (wins[a] - wins[b]).var(1, ddof=1)
    return {"wp": wp, "diff_var": dvar, "n_sims": settings.n_sims}


def _simulate_chunk(model, dec, entry_state, slot_lg, n_have, opp_logit, rule, st, rng):
    D, N = len(dec), st.n_sims
    t_home = dec["fld_is_home"].to_numpy(int).astype(bool)      # T is the home team
    era = (dec["season"].to_numpy(int) >= 2020).astype(float)
    inn0 = dec["entry_inning"].to_numpy(int)
    pos0 = 2 * (inn0 - 1) + np.where(t_home, 0, 1)             # half position: top=even, bottom=odd
    m = np.repeat(dec["entry_margin"].to_numpy(int)[:, None], N, 1)  # T margin
    done = np.zeros((D, N), bool)
    win = np.zeros((D, N))
    reserve_used = np.zeros((D, N), bool)
    depth_ptr = np.full((D, N), 2)
    rows = np.arange(D)[:, None]
    tail_cdf = model.tail_cdf
    fallback = slot_lg.shape[1] - 1                             # league-average reliever
    min_inn, lo, hi = rule
    step = 0
    while True:
        pos = pos0 + step
        inning = pos // 2 + 1
        bottom = (pos % 2).astype(bool)
        t_fields = bottom != t_home                             # T fields: top if home, bottom if away
        fld_home = ~bottom
        if step == 0:
            state = entry_state
        else:
            state = np.where((inning >= 10) & (era > 0), S_GHOST, S_EMPTY)
        ctx = model.context_logit(inning, state, fld_home.astype(float), era)       # (D, 4)
        fld_margin = np.where(t_fields[:, None], m, -m)
        logit = ctx[:, None, :] + model.t_margin[model.margin_index(fld_margin)]       # (D, N, 4)

        # pitcher on the mound
        if step == 0:
            slot = np.zeros((D, N), int)
        else:
            use_res = (~reserve_used & (inning[:, None] >= min_inn) & (m >= lo) & (m <= hi))
            depth_slot = np.where(depth_ptr < n_have[:, None], depth_ptr, fallback)
            slot = np.where(use_res, 1, depth_slot)
            active_t = t_fields[:, None] & ~done
            reserve_used |= use_res & active_t
            depth_ptr = np.where(active_t & ~use_res, depth_ptr + 1, depth_ptr)
        opp = opp_logit[np.clip(-m, -OPP_MARGIN_CAP, OPP_MARGIN_CAP) + OPP_MARGIN_CAP]   # opp fields: margin -m
        p_logit = np.where(t_fields[:, None, None], slot_lg[rows, slot], opp)
        logit = logit + p_logit
        logit -= logit.max(-1, keepdims=True)
        prob = np.exp(logit)
        cum = np.cumsum(prob, -1)
        u = rng.random((D, N)) * cum[..., -1]
        u2 = rng.random((D, N))
        cls = (u > cum[..., 0]).astype(int) + (u > cum[..., 1]) + (u > cum[..., 2])
        runs = np.where(cls == 3, 3 + np.searchsorted(tail_cdf, u2, side="right").clip(0, len(tail_cdf) - 1), cls)
        runs = np.where(done, 0, runs)
        m = np.where(t_fields[:, None], m - runs, m + runs)

        # end-of-game checks
        home_lead = np.where(t_home[:, None], m, -m)
        late = inning[:, None] >= 9
        end_top = ~bottom[:, None] & late & (home_lead > 0)
        end_bot = bottom[:, None] & late & (m != 0)
        new_end = ~done & (end_top | end_bot)
        win = np.where(new_end, (m > 0).astype(float), win)
        done |= new_end
        cap = bottom[:, None] & (inning[:, None] >= st.max_inning) & ~done
        win = np.where(cap, 0.5, win)
        done |= cap
        if done.all():
            return win
        step += 1
