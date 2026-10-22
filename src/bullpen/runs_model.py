"""Y1b outcome model: distribution of runs in the rest of the half-inning (0 / 1 / 2 / 3+).

This is the engine of the decision simulator (docs/SPEC.md §3, §4.1). It uses the same
population and feature encoding as the Y1 model ladder (bullpen/model_y1.py):
training-season states with a reliever on the mound, features fixed before fitting.

  M0  empirical 0/1/2/3+ distribution by base-out state
  M1  multinomial logistic, game state only
  M2  M1 + pitcher quality + usage   <- used by the simulator

"3+" is expanded to an exact run count with the empirical training distribution of
runs given >= 3 (capped at TAIL_MAX). Optional recalibration: three class-intercept
offsets fitted on validation relief states with every coefficient held fixed
(the rule-change drift correction allowed by SPEC §4.3, decided before the test set).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax

from .config import Config
from .model_y1 import (BASES, FEATURES, INNINGS, MARGIN_CAP, MARGINS, READ_COLS, STATE24,
                       make_model, prepare)

N_CLASS = 4
TAIL_MAX = 15
NUM_CLIPS = {"bf_d1": (0, 30), "apps_prev4": (0, 4), "pitcher_bf_in_game": (0, 12)}
STATE_COLS = sorted(set(READ_COLS) | {"runs_rest_half"})


# ------------------------------------------------------------------- I/O ------
def read_table(path: str, columns=None, splits=None, seasons=None, str_cols=()) -> pd.DataFrame:
    """Read a processed table (parquet or csv.gz), keeping only the requested splits/seasons.

    Test-season rows are dropped here, at read time, unless the caller asks for them.
    """
    if path.endswith(".parquet"):
        filters = []
        if splits is not None:
            filters.append(("split", "in", list(splits)))
        if seasons is not None:
            filters.append(("season", "in", list(seasons)))
        df = pd.read_parquet(path, columns=columns, filters=filters or None)
    else:
        df = pd.read_csv(path, usecols=columns, dtype={c: str for c in str_cols})
        if splits is not None:
            df = df[df["split"].isin(splits)]
        if seasons is not None:
            df = df[df["season"].isin(seasons)]
    for c in str_cols:
        if c in df:
            df[c] = df[c].astype(str)
    return df.reset_index(drop=True)


def load_relief_states(cfg: Config, splits=("train", "validation")) -> pd.DataFrame:
    if "test" in splits:
        raise ValueError("relief states for the outcome model are never read from the test split")
    manifest = json.loads((Path(cfg.out_dir) / "manifest.json").read_text())
    s = read_table(manifest["outputs"]["states"], STATE_COLS, splits=splits, str_cols=("state24",))
    assert set(s["split"]) <= set(splits)
    return s[s["pitcher_is_starter"] == 0].reset_index(drop=True)


def run_class(runs) -> np.ndarray:
    return np.clip(np.asarray(runs, dtype=int), 0, N_CLASS - 1)


# ----------------------------------------------------------------- model ------
@dataclass
class RunModel:
    """Fitted multinomial model stored as additive logit lookup tables (fast to simulate)."""
    name: str
    b0: np.ndarray                      # (4,) intercept incl. folded-in standardisation offsets
    t_inning: np.ndarray                # (10, 4)  index: inning 1..9 -> 0..8, 10+ -> 9
    t_state: np.ndarray                 # (24, 4)  index: STATE24 order
    t_margin: np.ndarray                # (11, 4)  index: margin + 5
    beta: dict                          # numeric feature -> (4,) coefficient per raw unit
    tail_cdf: np.ndarray                # P(runs <= 3 + j | runs >= 3), j = 0..TAIL_MAX-3
    recal: np.ndarray = field(default_factory=lambda: np.zeros(N_CLASS))
    meta: dict = field(default_factory=dict)

    # -------------------------------------------------------------- fitting --
    @classmethod
    def fit(cls, train: pd.DataFrame, name: str = "M2", weights=None) -> "RunModel":
        y = run_class(train["runs_rest_half"])
        X = prepare(train)
        pipe = make_model(name)
        pipe.set_params(lr__max_iter=5000)
        pipe.fit(X, y, lr__sample_weight=weights)
        if list(pipe.classes_) != list(range(N_CLASS)):
            raise ValueError(f"training data lacks some run classes: {pipe.classes_}")
        enc, lr = pipe.named_steps["enc"], pipe.named_steps["lr"]
        cats, nums = FEATURES[name]
        coef = lr.coef_.T                                  # (p, 4)
        b0 = lr.intercept_.copy()
        n_cat = sum(len(v) - 1 for v in cats.values())
        tables, pos = {}, 0
        for feat, levels in cats.items():
            t = np.zeros((len(levels), N_CLASS))
            t[1:] = coef[pos:pos + len(levels) - 1]       # first level dropped -> 0
            tables[feat] = t
            pos += len(levels) - 1
        scaler = enc.named_transformers_["num"]
        beta = {}
        for j, f in enumerate(nums):
            c = coef[n_cat + j] / scaler.scale_[j]
            beta[f] = c
            b0 = b0 - c * scaler.mean_[j]
        r = np.asarray(train["runs_rest_half"], dtype=int)
        w = np.ones(len(r)) if weights is None else np.asarray(weights, float)
        big = r >= 3
        tail = np.bincount(np.clip(r[big], 3, TAIL_MAX) - 3, weights=w[big], minlength=TAIL_MAX - 2)
        model = cls(name=name, b0=b0, t_inning=tables["inning_cat"], t_state=tables["state24"],
                    t_margin=tables["margin_cat"], beta=beta, tail_cdf=np.cumsum(tail) / tail.sum(),
                    meta={"n_train": int(len(train)), "n_iter": int(lr.n_iter_[0])})
        model._pipe = pipe
        return model

    # ------------------------------------------------------------- logits ----
    @staticmethod
    def inning_index(inning) -> np.ndarray:
        return np.minimum(np.asarray(inning, dtype=int), 10) - 1

    @staticmethod
    def state_index(state24) -> np.ndarray:
        lut = {s: i for i, s in enumerate(STATE24)}
        return np.array([lut[s] for s in np.asarray(state24, dtype=str)])

    @staticmethod
    def margin_index(margin) -> np.ndarray:
        return np.clip(np.asarray(margin, dtype=int), -MARGIN_CAP, MARGIN_CAP) + MARGIN_CAP

    def pitcher_logit(self, p: pd.DataFrame) -> np.ndarray:
        """(n, 4) logit contribution of pitcher features in raw units.

        p columns: runs_saved_per100, rating_eff_bf, pitched_d1, pitched_d2, bf_d1,
        apps_prev4, bf_in_game.
        """
        out = np.zeros((len(p), N_CLASS))
        if "pitcher_runs_saved_per100" not in self.beta:
            return out
        x = {
            "pitcher_runs_saved_per100": p["runs_saved_per100"].to_numpy(float),
            "log_rating_eff_bf": np.log1p(p["rating_eff_bf"].to_numpy(float)),
            "pitched_d1": p["pitched_d1"].to_numpy(float),
            "pitched_d2": p["pitched_d2"].to_numpy(float),
            "bf_d1": np.clip(p["bf_d1"].to_numpy(float), *NUM_CLIPS["bf_d1"]),
            "apps_prev4": np.clip(p["apps_prev4"].to_numpy(float), *NUM_CLIPS["apps_prev4"]),
            "pitcher_bf_in_game": np.clip(p["bf_in_game"].to_numpy(float), *NUM_CLIPS["pitcher_bf_in_game"]),
        }
        for f, v in x.items():
            out += v[:, None] * self.beta[f][None, :]
        return out

    def context_logit(self, inning, state24_idx, fld_is_home, era) -> np.ndarray:
        """(n, 4) logit from everything except margin and pitcher."""
        return (self.b0 + self.recal + self.t_inning[self.inning_index(inning)]
                + self.t_state[state24_idx]
                + np.asarray(fld_is_home, float)[:, None] * self.beta["fld_is_home"]
                + np.asarray(era, float)[:, None] * self.beta["extra_runner_era"])

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Class probabilities for state rows (states-table column names)."""
        logit = self.context_logit(df["inning"], self.state_index(df["state24"]),
                                   df["fld_is_home"], df["extra_runner_era"])
        logit = logit + self.t_margin[self.margin_index(df["fld_margin"])]
        if "pitcher_runs_saved_per100" in self.beta:
            p = pd.DataFrame({"runs_saved_per100": df["pitcher_runs_saved_per100"],
                              "rating_eff_bf": df["pitcher_rating_eff_bf"],
                              "pitched_d1": df["pitched_d1"], "pitched_d2": df["pitched_d2"],
                              "bf_d1": df["bf_d1"], "apps_prev4": df["apps_prev4"],
                              "bf_in_game": df["pitcher_bf_in_game"]})
            logit = logit + self.pitcher_logit(p)
        return softmax(logit, axis=1)

    # -------------------------------------------------------- recalibration ---
    def fit_recalibration(self, val: pd.DataFrame) -> np.ndarray:
        """Class-intercept offsets (class 0 fixed at 0) fitted on validation, coefficients fixed."""
        self.recal = np.zeros(N_CLASS)
        base = np.log(np.clip(self.predict_proba(val), 1e-12, 1))
        y = run_class(val["runs_rest_half"])

        def nll(d):
            z = base + np.r_[0.0, d]
            return -log_softmax(z, axis=1)[np.arange(len(y)), y].sum()

        res = minimize(nll, np.zeros(N_CLASS - 1), method="BFGS")
        self.recal = np.r_[0.0, res.x]
        return self.recal

    def to_dict(self) -> dict:
        return {"name": self.name, "b0": self.b0.tolist(), "recal": self.recal.tolist(),
                "t_inning": self.t_inning.tolist(), "t_state": self.t_state.tolist(),
                "t_margin": self.t_margin.tolist(), "beta": {k: v.tolist() for k, v in self.beta.items()},
                "tail_cdf": self.tail_cdf.tolist(), "meta": self.meta}


def empirical_m0(train: pd.DataFrame) -> pd.DataFrame:
    y = run_class(train["runs_rest_half"])
    tab = pd.crosstab(train["state24"].to_numpy(), y, normalize="index")
    return tab.reindex(columns=range(N_CLASS), fill_value=0.0)


def multiclass_ll(y, P) -> np.ndarray:
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))


def evaluate_ladder(val: pd.DataFrame, preds: dict, B: int, rng) -> dict:
    """Multiclass log loss, ranked probability score and expected-runs bias, game-clustered bootstrap."""
    y = run_class(val["runs_rest_half"])
    codes, uniq = pd.factorize(val["gid"])
    G = len(uniq)
    W = np.array([np.bincount(rng.integers(0, G, G), minlength=G) for _ in range(B)])
    n_g = np.bincount(codes, minlength=G).astype(float)
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    onehot_cum = (y[:, None] <= np.arange(N_CLASS - 1)[None, :]).astype(float)
    res, losses = {"n_rows": int(len(y)), "n_games": int(G),
                   "observed_share": np.bincount(y, minlength=N_CLASS).astype(float).__truediv__(len(y)).round(5).tolist()}, {}
    res["models"] = {}
    for name, P in preds.items():
        ll = multiclass_ll(y, P)
        rps = ((np.cumsum(P, 1)[:, :-1] - onehot_cum) ** 2).sum(1) / (N_CLASS - 1)
        losses[name] = ll
        L_g, R_g = np.bincount(codes, ll, G), np.bincount(codes, rps, G)
        res["models"][name] = {
            "log_loss": float(ll.mean()), "log_loss_ci": ci(W @ L_g / (W @ n_g)),
            "rps": float(rps.mean()), "rps_ci": ci(W @ R_g / (W @ n_g)),
            "pred_share": P.mean(0).round(5).tolist(),
        }
    names = list(preds)
    res["differences"] = {}
    for prev, cur in zip(names, names[1:]):
        D_g = np.bincount(codes, losses[cur] - losses[prev], G)
        boot = W @ D_g / (W @ n_g)
        res["differences"][f"{cur}-{prev}"] = {"log_loss_diff": float((losses[cur] - losses[prev]).mean()),
                                               "ci": ci(boot), "p_boot_not_better": float((boot >= 0).mean())}
    return res
