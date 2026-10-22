"""Single source of truth for every analysis choice.

Every threshold that defines "eligible", "available" or "highest-rated" lives
here, so a sensitivity analysis is just a different JSON file in configs/.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class Config:
    # ---- paths -----------------------------------------------------------
    raw_dir: str = "data/raw/retrosheet"      # data/raw/retrosheet/<year>/plays.csv
    out_dir: str = "data/processed"

    # ---- season splits (fixed before any modeling) -----------------------
    # Warm-up seasons are loaded ONLY so that early-2016 reliever ratings have
    # prior-season information. They never appear as modeling rows.
    warmup_seasons: tuple = (2014, 2015)
    train_seasons: tuple = (2016, 2017, 2018, 2019, 2020, 2021, 2022)
    val_seasons: tuple = (2023,)
    test_seasons: tuple = (2024, 2025)

    # ---- eligible decision points ----------------------------------------
    decision_innings: tuple = (7, 8, 9)
    # Score margin from the FIELDING (decision-making) team's point of view,
    # measured before the incoming reliever's first event. -2..+3 covers tied,
    # trailing by 1-2, and every save situation that starts an inning.
    margin_min: int = -2
    margin_max: int = 3
    # 2020-2021 regular-season doubleheader games were scheduled for 7 innings,
    # so "innings 7-9" means something different there; they are excluded.
    seven_inning_dh_seasons: tuple = (2020, 2021)
    regular_season_only: bool = True

    # ---- bullpen pool (who is on the staff as a reliever) ----------------
    pool_lookback_days: int = 30        # relief appearance for this team in last N days
    starter_exclusion_days: int = 4     # started a game in last N days -> not in bullpen
    position_player_min_pa: int = 100   # >= N batting PA (prior season + season-to-date)

    # ---- availability (rest) rules ----------------------------------------
    rule_back_to_back: bool = True      # pitched on BOTH d-1 and d-2 -> unavailable
    heavy_bf_prev_day: int = 10         # faced >= N batters on d-1 -> unavailable (0 = off)
    max_apps_prev4: int = 2             # > N appearances in d-4..d-1 -> unavailable
    rule_pitched_earlier_today: bool = True  # pitched in game 1 of a doubleheader
    min_available: int = 2              # need best + an alternative to compare

    # ---- reliever rating (run prevention, shrunk) ------------------------
    prior_season_weights: tuple = (0.6, 0.3)  # weight on season-1, season-2 totals
    shrink_k: float | None = None      # pseudo-BF of league-average; None = estimate on train
    shrink_k_bounds: tuple = (50.0, 3000.0)

    # ---------------------------------------------------------------------
    @property
    def model_seasons(self) -> tuple:
        return tuple(self.train_seasons) + tuple(self.val_seasons) + tuple(self.test_seasons)

    @property
    def all_seasons(self) -> tuple:
        return tuple(self.warmup_seasons) + self.model_seasons

    def split_of(self, season: int) -> str:
        if season in self.train_seasons:
            return "train"
        if season in self.val_seasons:
            return "validation"
        if season in self.test_seasons:
            return "test"
        if season in self.warmup_seasons:
            return "warmup"
        return "unused"

    # ---- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path | None) -> "Config":
        if path is None:
            return cls()
        raw = json.loads(Path(path).read_text())
        names = {f.name for f in fields(cls)}
        unknown = set(raw) - names
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        kw = {k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()}
        return cls(**kw)
