# Data

## Source

All game data come from Retrosheet <https://www.retrosheet.org/downloads/csvdownloads.html>.

## Coverage

| Season | Games | Events | Role | Notes |
|---|---|---|---|---|
| 2014 | 2,430 | 189,979 | warm-up | prior-season history for ratings only |
| 2015 | 2,429 | 189,588 | warm-up | prior-season history for ratings only |
| 2016 | 2,428 | 190,602 | train | |
| 2017 | 2,430 | 191,195 | train | |
| 2018 | 2,429 | 190,909 | train | |
| 2019 | 2,429 | 192,025 | train | |
| 2020 | 898 | 68,599 | train | 60-game season; 111 seven-inning doubleheader games |
| 2021 | 2,429 | 187,210 | train | 121 seven-inning doubleheader games |
| 2022 | 2,430 | 187,365 | train | |
| 2023 | 2,430 | 190,416 | validation | first season with pitch clock, shift limits, larger bases |
| 2024 | 2,429 | 188,937 | test | evaluated once |
| 2025 | 2,430 | 189,311 | test | evaluated once |

Regular season only. "Events" counts play-by-play rows after Retrosheet "no play" rows are removed.

## Unit of analysis and sample

**Event state:** the game situation before one play. Used for the run-scoring models. Model population: states with a relief pitcher on the mound (492,753 train, 80,901 validation).
**Decision point:** a reliever entering a regular-season game scheduled for 9 innings, in innings 7–9, with the fielding team between −2 and +3 runs, in a game not ending tied, with at least two relievers available. There are 64,426 in total: 43,588 train, 6,908 validation and 13,930 test.

| Funnel step | Rows |
|---|---|
| Relief appearances, 2016–2025 | 150,307 |
| Entering in innings 7–9 | 105,758 |
| 9-inning games, not ending tied | 105,383 |
| Fielding margin −2 to +3 | 64,971 |
| At least 2 relievers available | 64,426 |

## Data dictionary

All tables are written to `data/processed/`. `fld_` refers to the fielding (deciding) team and `bat_` to the batting team. Pitcher features are computed from games on earlier dates, or from earlier plays of the same game, and never from the current play or later.
