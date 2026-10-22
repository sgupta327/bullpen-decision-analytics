# When Should You Deploy Your Best Reliever? — Research Report
Sonia Gupta

## 1. Summary

This project examines whether a baseball team should use its highest-rated available reliever immediately in a close, late-inning situation or use the second-best reliever while saving the best for a later high-leverage opportunity.

Using Retrosheet play-by-play data and a simulation calibrated on historical MLB games, I compared these two strategies at 6,908 late-inning pitching changes in close 2023 games. The simulation estimates that using the best-rated available reliever immediately increases the deciding team's win probability by **0.112 percentage points on average** (95% interval incorporating game resampling and model-parameter uncertainty: **+0.095 to +0.142 percentage points**).

The estimated benefit is concentrated in the ninth inning, when the team has a narrow lead, and when the best reliever is clearly better than the alternative. The effect is small overall, and the results are model-based estimates rather than proof of a causal effect. This report presents validation-set results for 2023; the 2024–2025 held-out test seasons are not reported here.

## 2. Question and Strategy

**Question.** At a late-inning pitching change, should the team use its highest-rated available reliever now, or use the next-best reliever and save the best one for a later, higher-leverage situation?

**Strategies** (for the deciding team T at decision point t, with available relievers ranked by rating):

- **A, use now:** the rank 1 reliever enters now and rank 2 is held in reserve.
- **B, save:** the rank 2 reliever enters now and rank 1 is held in reserve.

Under Strategy B, the reserved reliever enters the first later half-inning beginning in the eighth inning or later when the team is tied or leading by 1–3 runs. Other later half-innings are assigned to the next available relievers in rating order, followed by a league-average reliever.

The main quantity of interest is the average difference in simulated win probability:

ΔWP = P(win | A) − P(win | B) 

Because the strategies differ in the timing of the top-rated reliever, this comparison estimates the value of using that reliever now rather than saving him under the specified reserve rule.

## 3. Data

Retrosheet play-by-play and game-information files for the 2014–2025 MLB regular seasons. Data sources, coverage, and the data dictionary can be found in [`DATA.md`](DATA.md).

| Split | Seasons | Role |
|---|---|---|
| Warm-up | 2014–2015 | historical information for pitcher ratings |
| Train | 2016–2022 | fit run-expectancy and outcome models |
| Validation | 2023 | model selection, calibration, reported results |
| Test | 2024–2025 | reserved for final evaluation (results are not included in this report) |

A decision point is a reliever entering a regular-season game scheduled for 9 innings, in the 7th–9th inning, with the fielding team between −2 and +3 runs, in a game not ending tied, and with at least two relievers available.

## 4. Definitions

Pitcher quality is estimated from historical run values using RE24, which accounts for runs scored and changes in expected runs across base-out states. Ratings are shrunk toward the league average so that small samples of past performance do not dominate a pitcher's estimated quality. Only information available before the decision date is used.

Because public data do not fully capture bullpen readiness, availability is approximated from recent team usage. The rule considers recent appearances, workload, and whether the pitcher has already appeared in the current game. It marks 3.4% of relievers who actually entered games as unavailable; stricter and more lenient versions are examined as sensitivity checks.

Full rating and availability specifications are documented in the project files.

## 5. Methods

I compare three models:

| Model | Features |
|---|---|
| M0 | empirical baseline |
| M1 | game-state baseline |
| M2 | pitcher-informed model |

The models predict whether a run scores in the remainder of a half-inning. A second model estimates the distribution of runs scored, which drives the game simulator. The simulator projects the remainder of each game under Strategies A and B, using common random numbers to reduce simulation noise. Each strategy is simulated 2,000 times per decision point.

The models are trained on 2016–2022 and evaluated on 2023. Because scoring in 2023 was higher than in the training period, the primary run model is recalibrated using 2023 data before the strategy comparison. Results without this recalibration are also checked.

Predictive performance is assessed using log loss and Brier score, alongside calibration measures. Uncertainty is estimated using game-clustered bootstrap resampling; a combined interval also incorporates variation from refitting the run model on bootstrap samples of the training data. The main strategy comparison is reported as a model-based estimate, not a causal claim.

## 6. Results (validation, 2023)

### 6.1 Predictive Performance

All relief states: 80,901 states in 2,429 games; observed rate 0.2744.

| Model | Log loss | Brier|
|---|---|---|
| M0 | 0.5176 [0.5114, 0.5236] | 0.1709 |
| M1 | 0.5170 [0.5107, 0.5230] | 0.1707 |
| M2 | 0.5165 [0.5103, 0.5227] | 0.1705 |

The improvement from M1 to M2 is statistically distinguishable from zero across all relief states, although the difference is small. At the narrower set of strategy decision points, the confidence interval for this improvement includes zero. This suggests pitcher features add useful information overall, but their incremental predictive value at the selected decision points is less certain.

The simulator's win-probability predictions also perform modestly better than a simple win-expectancy table on the 2023 validation data. These validation results do not replace evaluation on the held-out 2024–2025 seasons.


### 6.2 Strategy comparison

| Quantity | Value |
|---|---|
| Mean ΔWP | **+0.112 pp** |
| 95% CI, game resampling | [+0.103, +0.120] |
| 95% CI, combined with parameter uncertainty | [+0.095, +0.142] |
| Monte Carlo SE of the mean | 0.0022 pp |
| Decision points with ΔWP > 0 | 52.3% |
| Mean simulated WP: A / B / actual choice / observed | 0.5668 / 0.5657 / 0.5653 / 0.5618 |

The estimated effect is small on average, but it varies by game situation.

| Subgroup | n | ΔWP |
|---|---|---|
| Inning 7 | 2,421 | +0.022 |
| Inning 8 | 2,560 | +0.044 |
| Inning 9 | 1,927 | +0.314 |
| Trailing 1–2 | 1,952 | +0.028 |
| Tied | 1,405 | +0.025 |
| Leading 1 | 1,460 | +0.242 |
| Leading 2–3 | 2,091 | +0.156 |
| Bases empty | 5,549 | +0.109 |
| Runners on | 1,359 | +0.124 |
| Small rating gap (< 0.35) |
| Middle gap | 2,081 | +0.074 |
| Large gap (> 0.92) | 2,807 | +0.205 |
| Home | 3,666 | +0.112 |
| Away | 3,242 | +0.112 |

The largest estimated benefit appears in the ninth inning and when the team is protecting a narrow lead. It is also larger when the top-rated reliever's estimated quality exceeds that of the next-best option by a wider margin. In the ninth inning, the reserve strategy often has fewer later opportunities to use the saved reliever, which helps explain the larger estimated effect.

At the overall average of +0.112 percentage points per decision, the estimated value is approximately one additional win per 900 comparable decisions. This is a rough scale illustration, not a forecast of actual wins for a particular team.

### 6.5 Assumption tests

| Variant | n | Mean ΔWP (pp) |
|---|---|---|
| **Primary** | 6,908 | +0.112 |
| No recalibration | 6,908 | +0.105 |
| Reserve enters only in a save spot (inning ≥ 9, lead 1–3) | 6,908 | +0.174 |
| Reserve enters in late close spots (inning ≥ 8, margin −1..+1) | 6,908 | +0.157 |
| Training without 2020 | 6,908 | +0.109 |
| Stronger opponent bullpen (+1 run saved / 100 BF) | 6,908 | +0.110 |
| Strict availability rule | 6,840 | +0.123 |
| Lenient availability rule | 6,909 | +0.105 |
| Placebo: no pitcher features (M1) | 6,908 | 0.000 |

Alternative reserve rules also produce positive estimates. These results indicate that the main finding is not driven solely by the recalibration or one availability definition, although all estimates remain dependent on the model and assumptions.

A descriptive comparison of actual 2023 outcomes shows higher observed win rates when teams used their best available reliever than when they used other relievers. Those raw differences are confounded: managers tend to deploy their best relievers in situations where they are protecting leads. For that reason, actual win rates by pitcher choice are not interpreted as causal strategy effects.


## 7. Limitations

* **Approximate availability**: Public data do not show injuries, illness, warm-up status, pitch counts in full context, or all information available to managers.
* **Model dependence**: The simulation relies on the run model, simplified game rules, and the assumption that opponent relievers are league-average for the current margin.
* **Manager selection**: The outcome model learns from historical usage, which reflects managerial decisions. Simulation reduces reliance on raw outcome comparisons but does not remove all selection effects.
* **Limited scope**: The analysis considers actual pitching-change events in the seventh through ninth innings, rather than every inning in which a reliever could remain in the game.
* **Not causal proof**: The estimated strategy difference is conditional on the model and reserve rule. It does not establish that adopting the strategy would cause the same change in real-world wins.