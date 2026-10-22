"""Markdown report + figures for the decision simulation (reads reports/sim/metrics_<split>.json)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
DIV_NEG, DIV_MID, DIV_POS = "#e34948", "#f0efec", "#2a78d6"     # B better <-> A better
ORDER = {"inning": ["7", "8", "9"], "margin": ["trailing 1-2", "tied", "leading 1", "leading 2-3"],
         "bases": ["bases empty", "runners on"], "gap_tercile": ["small gap", "middle gap", "large gap"],
         "home": ["home", "away"]}
TITLES = {"inning": "Inning", "margin": "Score at entry", "bases": "Base state",
          "gap_tercile": "Best-vs-alternative rating gap", "home": "Deciding team"}


def pp(x, nd=2):
    return f"{100 * x:+.{nd}f}"


def ci_pp(c, nd=2):
    return f"[{100 * c[0]:+.{nd}f}, {100 * c[1]:+.{nd}f}]"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def plot_calibration(m, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sc = m["self_consistency"]
    fig, ax = plt.subplots(figsize=(6.2, 5.6), facecolor=SURFACE)
    _style(ax)
    series = [("WE0_table", sc[m["meta"]["primary"]], INK2, "Empirical win-expectancy table"),
              ("SIM_actual", sc.get("no_recal"), ORANGE, "Simulator, no recalibration"),
              ("SIM_actual", sc.get("recal"), BLUE, "Simulator, 2023 recalibration")]
    for key, block, color, label in series:
        if block is None:
            continue
        r = block["models"][key]["reliability"]
        mm = block["models"][key]
        ax.plot(r["p_mean"], r["y_mean"], "-", color=color, lw=2, zorder=3)
        ax.plot(r["p_mean"], r["y_mean"], "o", ms=6, color=color, mec=SURFACE, mew=2, zorder=4,
                label=f"{label}  (slope {mm['cal_slope']:.2f}, log loss {mm['log_loss']:.4f})")
    ax.plot([0, 1], [0, 1], ":", color=INK2, lw=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted win probability (team's actual choice)", color=INK)
    ax.set_ylabel("Observed win rate", color=INK)
    ax.set_title(f"Simulator self-consistency, {m['meta']['split']} decision points", color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def plot_subgroups(m, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [("All decision points", m["dwp"]["overall"] | {"combined_ci": m["dwp"]["parameter_uncertainty"]["combined_ci"]}, True)]
    for g, levels in ORDER.items():
        rows.append((TITLES[g], None, True))
        for lv in levels:
            if lv in m["dwp"]["subgroups"][g]:
                rows.append((f"   {lv}", m["dwp"]["subgroups"][g][lv], False))
    fig, ax = plt.subplots(figsize=(7.4, 0.32 * len(rows) + 1.2), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    yy = np.arange(len(rows))[::-1]
    for y, (label, r, head) in zip(yy, rows):
        if r is None:
            continue
        ax.plot([100 * r["combined_ci"][0], 100 * r["combined_ci"][1]], [y, y], color=BLUE, lw=1.2, alpha=0.55)
        ax.plot([100 * r["ci"][0], 100 * r["ci"][1]], [y, y], color=BLUE, lw=3.2, solid_capstyle="round")
        ax.plot(100 * r["mean"], y, "o", ms=7, color=BLUE, mec=SURFACE, mew=2, zorder=4)
        ax.text(1.01, y, f"{pp(r['mean'])}  (n={r['n']:,})", transform=ax.get_yaxis_transform(),
                va="center", fontsize=8, color=INK2)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(yy)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9, color=INK)
    for t, (_, r, head) in zip(ax.get_yticklabels(), rows):
        if r is None or head:
            t.set_fontweight("bold")
    ax.set_xlabel("ΔWP = P(win | best now) − P(win | save best), percentage points\n"
                  "thick: game-resampling 95% interval · thin: + model-parameter uncertainty", color=INK, fontsize=9)
    ax.set_title(f"Value of using the best reliever now ({m['meta']['split']})", color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_grid(m, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cells = {(r["inning"], r["margin"]): r for r in m["dwp"]["inning_x_margin"]}
    inn, mar = ORDER["inning"], ORDER["margin"]
    Z = np.array([[100 * cells[(i, j)]["mean"] if (i, j) in cells else np.nan for j in mar] for i in inn])
    lim = np.nanmax(np.abs(Z)) or 1
    cmap = LinearSegmentedColormap.from_list("div", [DIV_NEG, DIV_MID, DIV_POS])
    fig, ax = plt.subplots(figsize=(7.2, 3.6), facecolor=SURFACE)
    ax.imshow(Z, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for a in range(len(inn)):
        for b in range(len(mar)):
            c = cells.get((inn[a], mar[b]))
            if c:
                ax.text(b, a, f"{pp(c['mean'] , 2)}\nn={c['count']:,}", ha="center", va="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(mar)), mar, color=INK, fontsize=9)
    ax.set_yticks(range(len(inn)), [f"inning {i}" for i in inn], color=INK, fontsize=9)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(mar)), minor=True); ax.set_yticks(np.arange(-.5, len(inn)), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=3)
    ax.set_title("Mean ΔWP (pp) by inning and score — blue: use best now, red: save him (exploratory)",
                 color=INK, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def write_report(out: Path, split: str):
    out = Path(out)
    m = json.loads((out / f"metrics_{split}.json").read_text())
    plot_calibration(m, out / f"calibration_{split}.png")
    plot_subgroups(m, out / f"dwp_subgroups_{split}.png")
    plot_grid(m, out / f"dwp_grid_{split}.png")
    o, pu, meta = m["dwp"]["overall"], m["dwp"]["parameter_uncertainty"], m["meta"]
    L = [f"# Decision simulation: best reliever now vs save him ({split})", "",
         f"{meta['n_decisions']:,} decision points in {meta['n_games']:,} games. Primary run model: "
         f"{'M2 with 2023 class-intercept recalibration' if meta['primary'] == 'recal' else 'M2, no recalibration'}. "
         f"Intervals: {meta['boot']}-resample game-clustered bootstrap; 'combined' adds run-model parameter "
         f"uncertainty from {pu['refits']} refits on game-bootstrap resamples of 2016–2022.", "",
         "## Headline", "",
         "| Quantity | Value |", "|---|---|",
         f"| Mean ΔWP = P(win \\| A) − P(win \\| B) | **{pp(o['mean'], 3)} pp** |",
         f"| 95% interval, game resampling | {ci_pp(o['ci'], 3)} pp |",
         f"| 95% interval, combined with parameter uncertainty | {ci_pp(pu['combined_ci'], 3)} pp |",
         f"| Monte Carlo SE of the mean | {100 * o['mc_se']:.4f} pp |",
         f"| Monte Carlo SE per decision point (median / max) | {100 * m['dwp']['mc_se_per_decision_median']:.3f} / "
         f"{100 * m['dwp']['mc_se_per_decision_max']:.3f} pp |",
         f"| Share of decision points with ΔWP > 0 | {o['share_positive']:.1%} |",
         f"| Mean simulated WP: A / B / team's actual choice / observed | {m['dwp']['wp_A']:.4f} / {m['dwp']['wp_B']:.4f} / "
         f"{m['dwp']['wp_actual']:.4f} / {m['dwp']['observed_win']:.4f} |", "",
         f"![subgroups](dwp_subgroups_{split}.png)", "",
         "## ΔWP by pre-specified subgroup", "",
         "| Subgroup | n | Mean ΔWP (pp) | 95% CI, games | 95% CI, combined | MC SE (pp) |", "|---|---|---|---|---|---|"]
    for g, levels in ORDER.items():
        for lv in levels:
            r = m["dwp"]["subgroups"][g].get(lv)
            if r:
                L.append(f"| {TITLES[g]}: {lv} | {r['n']:,} | {pp(r['mean'], 3)} | {ci_pp(r['ci'], 3)} | "
                         f"{ci_pp(r['combined_ci'], 3)} | {100 * r['mc_se']:.4f} |")
    L += ["", f"Rating-gap tercile cut points come from training decision points: "
          f"{meta['gap_cuts_from_train'][0]:.2f} and {meta['gap_cuts_from_train'][1]:.2f} runs saved per 100 BF.", "",
          "### Inning × score (exploratory, not pre-specified)", "", f"![grid](dwp_grid_{split}.png)", ""]

    L += ["## Simulator self-consistency (team's actual choice vs realised wins)", "",
          "| Run model | Predictor | Log loss | Brier | Cal. intercept | Cal. slope | ECE |", "|---|---|---|---|---|---|---|"]
    for tag, block in m["self_consistency"].items():
        for name, mm in block["models"].items():
            if name == "WE0_table" and tag != meta["primary"]:
                continue
            f = lambda k: f"{mm[k]:.4f} [{mm[k + '_ci'][0]:.4f}, {mm[k + '_ci'][1]:.4f}]"
            L.append(f"| {tag if name != 'WE0_table' else '—'} | {name} | {f('log_loss')} | {f('brier')} | "
                     f"{f('cal_intercept')} | {f('cal_slope')} | {mm['ece']:.4f} |")
        dd = block["differences"]["SIM_actual-WE0_table"]
        L.append(f"| {tag} | SIM − WE0 log loss | {dd['log_loss_diff']:+.5f} [{dd['ci'][0]:+.5f}, {dd['ci'][1]:+.5f}] | | | | |")
    L += ["", "WE0_table: training win rate by inning × home × margin at the decision point (add-one smoothed).",
          "", f"![calibration](calibration_{split}.png)", ""]

    if "y1b_ladder" in m:
        y = m["y1b_ladder"]
        L += ["## Run-distribution model (Y1b: 0 / 1 / 2 / 3+ runs, rest of half-inning)", "",
              f"{y['n_rows']:,} validation relief states, {y['n_games']:,} games. Observed shares {y['observed_share']}.", "",
              "| Model | Log loss | RPS | Mean predicted shares |", "|---|---|---|---|"]
        for n, mm in y["models"].items():
            L.append(f"| {n} | {mm['log_loss']:.4f} [{mm['log_loss_ci'][0]:.4f}, {mm['log_loss_ci'][1]:.4f}] | "
                     f"{mm['rps']:.4f} | {mm['pred_share']} |")
        L += ["", "| Comparison | Δ log loss | 95% CI | Share of resamples not better |", "|---|---|---|---|"]
        for k, dd in y["differences"].items():
            L.append(f"| {k} | {dd['log_loss_diff']:+.5f} | [{dd['ci'][0]:+.5f}, {dd['ci'][1]:+.5f}] | {dd['p_boot_not_better']:.3f} |")
        L += ["", f"2023 recalibration offsets (classes 1, 2, 3+ vs 0): {[round(v, 4) for v in y['M2_recal_offsets'][1:]]}; "
              f"predicted shares after recalibration {y['M2_recal_pred_share']}.", ""]

    L += ["## Parameter uncertainty", "",
          f"Mean ΔWP across {pu['refits']} refits ({pu['n_sims_per_refit']} simulations each): SD {100 * pu['param_sd']:.3f} pp, "
          f"2.5–97.5% range {ci_pp(pu['param_interval'], 3)} pp. Combined with game resampling: {ci_pp(pu['combined_ci'], 3)} pp "
          f"(share ≤ 0: {pu['combined_p_le_0']:.3f}).", "",
          "## Assumption tests", "", "| Variant | n | Mean ΔWP (pp) | 95% CI, games | What changes |", "|---|---|---|---|---|",
          f"| **Primary** | {o['n']:,} | {pp(o['mean'], 3)} | {ci_pp(o['ci'], 3)} | — |"]
    for name, s in m["sensitivity"].items():
        if "mean" in s:
            L.append(f"| {name} | {s['n']:,} | {pp(s['mean'], 3)} | {ci_pp(s['ci'], 3)} | {s['description']} |")
        else:
            L.append(f"| {name} | — | — | — | {s['description']} |")
    L += ["", "## Descriptive only (confounded by manager choice)", "", "| Team actually used | Realised win rate | n |", "|---|---|---|"]
    for r in m["descriptive_actual_choice"]:
        L.append(f"| {r['actual_choice']} | {r['mean']:.3f} | {r['count']:,} |")
    L += ["", "## Reading this", "",
          "- ΔWP > 0 favours using the best available reliever now; ΔWP < 0 favours saving him under the pre-specified reserve rule.",
          "- The simulator works at the half-inning level: each reliever pitches the rest of the half-inning he starts; "
          "the opponent's pitcher is the average training reliever for its current margin; base-out state is reset each half-inning.",
          "- In the 9th, B's reserve can only appear in extra innings (the next half-inning starting in inning 8+ with T "
          "tied or ahead), so most of the 9th-inning ΔWP is 'use the best reliever rather than not at all'.",
          "- Effect sizes are bounded by how much the shrunk RE24 rating predicts runs (Y1b M2 − M1 is small). If true "
          "talent gaps are larger than the rating gaps, ΔWP is understated (SPEC §4.4); the combined interval covers "
          "parameter noise, not that kind of misspecification.",
          "- The answer is \"best now vs this specific save policy\" (SPEC §4.4), with availability approximated by the rest rule.",
          ""]
    (out / f"report_{split}.md").write_text("\n".join(L))
