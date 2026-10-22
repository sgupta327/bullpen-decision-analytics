"""Y1 outcome model: P(batting team scores >= 1 run in the rest of the half-inning).

    python -m bullpen.model_y1 --config configs/default.json [--boot 1000]

Model ladder (docs/SPEC.md §4.1), fit on training-season states with a reliever on the mound:
  M0  empirical rate by base-out state
  M1  logistic, game state only: inning, base-out state, margin (capped ±5), home, rules era
  M2  M1 + pitcher quality (shrunk RE24 rating, its effective sample) + usage/fatigue
Scored on the validation seasons only. Test-season rows are filtered out at read time,
so nothing here can look at 2024-2025 outcomes.

Uncertainty: game-clustered bootstrap (resample games with replacement).
Writes reports/y1/{metrics.json, coefficients.csv, reliability.png} and reports/y1/report.md.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import Config

TARGET = "scored_rest_half"
MARGIN_CAP = 5
INNINGS = [str(i) for i in range(1, 10)] + ["10+"]
BASES = ["000", "100", "010", "001", "110", "101", "011", "111"]
STATE24 = [f"{o}_{b}" for o in range(3) for b in BASES]
MARGINS = list(range(-MARGIN_CAP, MARGIN_CAP + 1))

GAME_STATE_CAT = {"inning_cat": INNINGS, "state24": STATE24, "margin_cat": MARGINS}
GAME_STATE_NUM = ["fld_is_home", "extra_runner_era"]
# Historical pitcher quality + usage. All are computed from games strictly before
# this one (rating, usage) or from earlier events of this game (bf_in_game).
PITCHER_NUM = ["pitcher_runs_saved_per100", "log_rating_eff_bf", "pitched_d1", "pitched_d2",
               "bf_d1", "apps_prev4", "pitcher_bf_in_game"]
FEATURES = {"M1": (GAME_STATE_CAT, GAME_STATE_NUM),
            "M2": (GAME_STATE_CAT, GAME_STATE_NUM + PITCHER_NUM)}

# Known only at or after the event; must never be a model input.
POST_EVENT = {"runs", "re24", "runs_rest_half", "scored_rest_half", "half_truncated",
              "fld_win", "fld_runs_allowed_rest_game", "is_tie"}
RAW_INPUTS = {"inning", "state24", "fld_margin", "fld_is_home", "extra_runner_era",
              "pitcher_runs_saved_per100", "pitcher_rating_eff_bf", "pitched_d1", "pitched_d2",
              "bf_d1", "apps_prev4", "pitcher_bf_in_game"}
READ_COLS = sorted(RAW_INPUTS | {TARGET, "gid", "season", "split", "row_idx", "pitcher_is_starter"})


# ---------------------------------------------------------------- data --------
def _read(path: str, columns=None, splits=("train", "validation")) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path, columns=columns, filters=[("split", "in", list(splits))])
    df = pd.read_csv(path, usecols=columns)
    return df[df["split"].isin(splits)]


def load(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = Path(cfg.out_dir)
    manifest = json.loads((out / "manifest.json").read_text())
    states = _read(manifest["outputs"]["states"], READ_COLS)
    assert set(states["split"]) <= {"train", "validation"}, "test rows must never be loaded"
    states = states[states["pitcher_is_starter"] == 0].reset_index(drop=True)
    dec = _read(manifest["outputs"]["decisions"], ["decision_id", "entry_row", "split"])
    return states, dec


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    inn = df["inning"].astype(int)
    X["inning_cat"] = np.where(inn >= 10, "10+", inn.astype(str))
    X["state24"] = df["state24"].astype(str)
    X["margin_cat"] = df["fld_margin"].clip(-MARGIN_CAP, MARGIN_CAP).astype(int)
    X["fld_is_home"] = df["fld_is_home"].astype(float)
    X["extra_runner_era"] = df["extra_runner_era"].astype(float)
    X["pitcher_runs_saved_per100"] = df["pitcher_runs_saved_per100"].astype(float)
    X["log_rating_eff_bf"] = np.log1p(df["pitcher_rating_eff_bf"].astype(float))
    X["pitched_d1"] = df["pitched_d1"].astype(float)
    X["pitched_d2"] = df["pitched_d2"].astype(float)
    X["bf_d1"] = df["bf_d1"].clip(0, 30).astype(float)
    X["apps_prev4"] = df["apps_prev4"].clip(0, 4).astype(float)
    X["pitcher_bf_in_game"] = df["pitcher_bf_in_game"].clip(0, 12).astype(float)
    return X


def make_model(name: str) -> Pipeline:
    cats, nums = FEATURES[name]
    enc = ColumnTransformer([
        ("cat", OneHotEncoder(categories=list(cats.values()), drop="first", sparse_output=False),
         list(cats)),
        ("num", StandardScaler(), nums),
    ])
    return Pipeline([("enc", enc), ("lr", LogisticRegression(C=np.inf, max_iter=2000))])


# ------------------------------------------------------------- metrics --------
def _ll(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def recalib(z, y, w, slope=True, start=(0.0, 1.0)):
    """Weighted logistic recalibration logit P(y) = a + b*z (b fixed at 1 if not slope)."""
    a, b = start if slope else (start[0], 1.0)
    for _ in range(50):
        p = expit(a + b * z)
        r, v = w * (y - p), w * p * (1 - p)
        if slope:
            vz = v * z
            H = np.array([[v.sum(), vz.sum()], [vz.sum(), (vz * z).sum()]])
            step = np.linalg.solve(H, np.array([r.sum(), (r * z).sum()]))
            a, b = a + step[0], b + step[1]
        else:
            step = np.array([r.sum() / v.sum()])
            a += step[0]
        if np.abs(step).max() < 1e-9:
            break
    return a, b


def evaluate(y, preds: dict, gid, B: int, rng, n_bins: int = 15) -> dict:
    """Point estimates + game-clustered bootstrap CIs for every model and model difference."""
    codes, uniq = pd.factorize(gid)
    G = len(uniq)
    W = np.array([np.bincount(rng.integers(0, G, G), minlength=G) for _ in range(B)]).reshape(B, G)
    n_g = np.bincount(codes, minlength=G).astype(float)
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if B else None
    res = {"n_rows": int(len(y)), "n_games": int(G), "base_rate": float(y.mean()), "models": {}}
    losses = {}
    for name, p in preds.items():
        ll, br = _ll(y, p), (p - y) ** 2
        losses[name] = ll
        L_g = np.bincount(codes, ll, G)
        Br_g = np.bincount(codes, br, G)
        boot_ll, boot_br = W @ L_g / (W @ n_g), W @ Br_g / (W @ n_g)

        z = logit(np.clip(p, 1e-6, 1 - 1e-6))
        ones = np.ones_like(y, dtype=float)
        a0, _ = recalib(z, y, ones, slope=False)
        a1, b1 = recalib(z, y, ones, slope=True)
        boot_int, boot_slope = np.empty(B), np.empty(B)
        for i in range(B):
            w = W[i][codes].astype(float)
            boot_int[i] = recalib(z, y, w, slope=False, start=(a0, 1.0))[0]
            boot_slope[i] = recalib(z, y, w, slope=True, start=(a1, b1))[1]

        # Expected calibration error on fixed equal-frequency bins.
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        k = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
        K = len(edges) - 1
        idx = codes * K + k
        Sy = np.bincount(idx, y, G * K).reshape(G, K)
        Sp = np.bincount(idx, p, G * K).reshape(G, K)
        Sn = np.bincount(idx, None, G * K).reshape(G, K)
        def ece(sy, sp, sn):
            sn_ = np.where(sn > 0, sn, 1)
            return (np.abs(sy / sn_ - sp / sn_) * sn).sum(-1) / sn.sum(-1)
        bins = pd.DataFrame({"p_mean": Sp.sum(0) / Sn.sum(0), "y_mean": Sy.sum(0) / Sn.sum(0),
                             "n": Sn.sum(0)})
        res["models"][name] = {
            "log_loss": float(ll.mean()), "log_loss_ci": ci(boot_ll),
            "brier": float(br.mean()), "brier_ci": ci(boot_br),
            "cal_intercept": float(a0), "cal_intercept_ci": ci(boot_int),
            "cal_slope": float(b1), "cal_slope_ci": ci(boot_slope),
            # ECE is biased upward under resampling, so only the point estimate is reported.
            "ece": float(ece(Sy.sum(0), Sp.sum(0), Sn.sum(0))),
            "mean_pred": float(p.mean()),
            "reliability": bins.round(5).to_dict("list"),
        }
    names = list(preds)
    res["differences"] = {}
    for prev, cur in zip(names, names[1:]):
        D_g = np.bincount(codes, losses[cur] - losses[prev], G)
        boot = W @ D_g / (W @ n_g)
        res["differences"][f"{cur}-{prev}"] = {
            "log_loss_diff": float((losses[cur] - losses[prev]).mean()), "ci": ci(boot),
            "p_boot_not_better": float((boot >= 0).mean()) if B else None,
        }
    return res


# ----------------------------------------------------------------- run --------
def fit_all(train: pd.DataFrame):
    m0 = train.groupby("state24")[TARGET].mean()
    Xtr, ytr = prepare(train), train[TARGET].to_numpy()
    models = {name: make_model(name).fit(Xtr, ytr) for name in FEATURES}
    return m0, models


def predict_all(df: pd.DataFrame, m0: pd.Series, models: dict) -> dict:
    X = prepare(df)
    preds = {"M0": df["state24"].map(m0).fillna(m0.mean()).to_numpy(dtype=float)}
    preds.update({n: m.predict_proba(X)[:, 1] for n, m in models.items()})
    return preds


def coefficients(models: dict) -> pd.DataFrame:
    rows = []
    for name, m in models.items():
        enc, lr = m.named_steps["enc"], m.named_steps["lr"]
        names = enc.get_feature_names_out()
        scale = np.ones(len(names))
        num = enc.named_transformers_["num"]
        n_num = len(num.scale_)
        scale[-n_num:] = num.scale_
        for f, c, s in zip(names, lr.coef_[0], scale):
            rows.append({"model": name, "feature": f, "coef_std": c, "coef_per_unit": c / s,
                         "odds_ratio_per_unit": np.exp(c / s)})
        rows.append({"model": name, "feature": "intercept", "coef_std": lr.intercept_[0],
                     "coef_per_unit": np.nan, "odds_ratio_per_unit": np.nan})
    return pd.DataFrame(rows)


def plot_reliability(results: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [k for k in results if k.startswith("validation")]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 5), squeeze=False)
    colors = {"M0": "#888888", "M1": "#2a6fdb", "M2": "#d9480f"}
    for ax, key in zip(axes[0], panels):
        lo = hi = None
        for name, m in results[key]["models"].items():
            r = m["reliability"]
            ax.plot(r["p_mean"], r["y_mean"], "o-", ms=4, color=colors[name],
                    label=f"{name}  slope {m['cal_slope']:.2f}, ECE {m['ece']:.4f}")
            lo = min(r["p_mean"] + r["y_mean"] + ([lo] if lo is not None else []))
            hi = max(r["p_mean"] + r["y_mean"] + ([hi] if hi is not None else []))
        ax.plot([lo, hi], [lo, hi], ":", color="black", lw=1)
        ax.set_title(f"{key.replace('_', ' ')}  (n={results[key]['n_rows']:,})")
        ax.set_xlabel("predicted P(run scores, rest of half)")
        ax.set_ylabel("observed rate")
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _fmt(m, k):
    if m.get(f"{k}_ci") is None:
        return f"{m[k]:.4f}"
    lo, hi = m[f"{k}_ci"]
    return f"{m[k]:.4f} [{lo:.4f}, {hi:.4f}]"


def write_report(results: dict, coefs: pd.DataFrame, meta: dict, path: Path):
    L = ["# Y1 model ladder: run scores in rest of half-inning", "",
         f"Built {meta['built']}. Fit on training seasons {meta['train_seasons']} "
         f"(relief states, n={meta['n_train']:,}); scored on validation {meta['val_seasons']}. "
         f"95% intervals from a {meta['boot']}-resample game-clustered bootstrap. "
         "Test seasons were not loaded.", ""]
    for key, r in results.items():
        L += [f"## {key.replace('_', ' ')}", "",
              f"{r['n_rows']:,} states in {r['n_games']:,} games; observed rate {r['base_rate']:.4f}.", "",
              "| Model | Log loss | Brier | Cal. intercept | Cal. slope | ECE | Mean pred |",
              "|---|---|---|---|---|---|---|"]
        for n, m in r["models"].items():
            L.append(f"| {n} | {_fmt(m, 'log_loss')} | {_fmt(m, 'brier')} | {_fmt(m, 'cal_intercept')} "
                     f"| {_fmt(m, 'cal_slope')} | {m['ece']:.4f} | {m['mean_pred']:.4f} |")
        if r.get("differences"):
            L += ["", "| Comparison | Δ log loss | 95% CI | Share of resamples not better |", "|---|---|---|---|"]
            for k, d in r["differences"].items():
                if d["ci"] is None:
                    L.append(f"| {k} | {d['log_loss_diff']:+.5f} | — | — |")
                else:
                    L.append(f"| {k} | {d['log_loss_diff']:+.5f} | [{d['ci'][0]:+.5f}, {d['ci'][1]:+.5f}] "
                             f"| {d['p_boot_not_better']:.3f} |")
        L.append("")
    m2 = coefs[(coefs["model"] == "M2") & coefs["feature"].str.startswith("num__")]
    L += ["## M2 non-game-state coefficients", "",
          "| Feature | Odds ratio per unit | Coef (standardised) |", "|---|---|---|"]
    for r in m2.itertuples():
        L.append(f"| {r.feature.removeprefix('num__')} | {r.odds_ratio_per_unit:.4f} | {r.coef_std:+.4f} |")
    L += ["", "![reliability](reliability.png)", ""]
    path.write_text("\n".join(L))


def run(cfg: Config, boot: int = 1000, out_dir: str = "reports/y1", seed: int = 20261022) -> dict:
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    states, dec = load(cfg)
    leaked = POST_EVENT & RAW_INPUTS
    assert not leaked, f"post-event columns used as features: {leaked}"

    train = states[states["split"] == "train"]
    val = states[states["split"] == "validation"]
    m0, models = fit_all(train)

    val_dec_rows = set(dec.loc[dec["split"] == "validation", "entry_row"])
    tr_dec_rows = set(dec.loc[dec["split"] == "train", "entry_row"])
    subsets = {
        "validation_all_relief": val,
        "validation_decision_points": val[val["row_idx"].isin(val_dec_rows)],
    }
    rng = np.random.default_rng(seed)
    results = {}
    for key, df in subsets.items():
        preds = predict_all(df, m0, models)
        results[key] = evaluate(df[TARGET].to_numpy(dtype=float), preds, df["gid"].to_numpy(), boot, rng)
    # In-sample reference (no bootstrap): a large train/validation gap would hint at overfit or leakage.
    for key, df in {"train_all_relief": train,
                    "train_decision_points": train[train["row_idx"].isin(tr_dec_rows)]}.items():
        preds = predict_all(df, m0, models)
        results[key] = evaluate(df[TARGET].to_numpy(dtype=float), preds, df["gid"].to_numpy(), 0, rng)

    coefs = coefficients(models)
    coefs.to_csv(out / "coefficients.csv", index=False)
    meta = {"built": time.strftime("%Y-%m-%d %H:%M"), "train_seasons": list(cfg.train_seasons),
            "val_seasons": list(cfg.val_seasons), "n_train": len(train), "boot": boot, "seed": seed,
            "features": {k: list(c) + n for k, (c, n) in FEATURES.items()},
            "seconds": round(time.time() - t0, 1)}
    (out / "metrics.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    plot_reliability(results, out / "reliability.png")
    write_report(results, coefs, meta, out / "report.md")
    return {"meta": meta, "results": results}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default="reports/y1")
    args = ap.parse_args(argv)
    r = run(Config.from_json(args.config), args.boot, args.out)
    for key in ("validation_all_relief", "validation_decision_points"):
        print(key, {n: round(m["log_loss"], 5) for n, m in r["results"][key]["models"].items()},
              r["results"][key]["differences"])


if __name__ == "__main__":
    main()
