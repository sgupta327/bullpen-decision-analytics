"""Load Retrosheet play-by-play CSVs into one normalised frame.

Supported inputs (one folder per season under cfg.raw_dir/<year>/):
  * Retrosheet's own CSV downloads (plays.csv, optionally gameinfo.csv)
  * Chadwick `cwevent` output generated from Retrosheet event files

Column names differ between those formats, so each canonical field lists the
source names it may appear under. The loader fails loudly, naming the missing
field and the columns it did find, rather than guessing.
"""
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = {
    "gid": ["gid", "game_id", "GAME_ID"],
    "inning": ["inning", "INN_CT"],
    "bat_home": ["top_bot", "bat_home", "vis_home", "BAT_HOME_ID"],  # 1 = home team batting
    "pitcher": ["pitcher", "PIT_ID", "RESP_PIT_ID"],
    "outs_pre": ["outs_pre", "OUTS_CT"],
    "br1_pre": ["br1_pre", "BASE1_RUN_ID"],
    "br2_pre": ["br2_pre", "BASE2_RUN_ID"],
    "br3_pre": ["br3_pre", "BASE3_RUN_ID"],
    "runs": ["runs", "EVENT_RUNS_CT"],
}
OPTIONAL = {
    "bat_team": ["batteam", "bat_team", "BAT_TEAM_ID"],
    "fld_team": ["pitteam", "fld_team", "FLD_TEAM_ID"],
    "vis_team": ["visteam", "AWAY_TEAM_ID", "away_team"],
    "batter": ["batter", "BAT_ID"],
    "is_pa": ["pa", "BAT_EVENT_FL"],
    "event": ["event", "EVENT_TX"],
    "date": ["date", "GAME_DT"],
    "gametype": ["gametype", "GAME_TYPE"],
}


def _pick(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _find_file(season_dir: Path, stem: str) -> Path | None:
    hits = sorted(p for p in season_dir.glob("*") if p.is_file()
                  and stem in p.name.lower() and p.suffix.lower() in (".csv", ".txt", ".gz"))
    return hits[0] if hits else None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _to_flag(s: pd.Series) -> pd.Series:
    m = s.astype(str).str.strip().str.upper().map({"T": 1, "F": 0, "TRUE": 1, "FALSE": 0})
    return m.fillna(pd.to_numeric(s, errors="coerce")).fillna(0).astype(int)


def load_season(year: int, raw_dir: str | Path, regular_season_only: bool = True):
    """Return (plays_df, file_manifest) for one season."""
    season_dir = Path(raw_dir) / str(year)
    plays_path = _find_file(season_dir, "plays") or _find_file(season_dir, "event")
    if plays_path is None:
        raise FileNotFoundError(
            f"No plays file for {year}. Expected e.g. {season_dir}/plays.csv "
            "(see README: 'Getting the data').")
    raw = pd.read_csv(plays_path, dtype=str, low_memory=False, keep_default_na=False)
    manifest = [{"season": year, "file": str(plays_path), "sha256": sha256(plays_path),
                 "rows": len(raw)}]

    out = pd.DataFrame(index=raw.index)
    for canon, cands in REQUIRED.items():
        col = _pick(raw, cands)
        if col is None:
            raise KeyError(f"{plays_path}: cannot find '{canon}' (tried {cands}). "
                           f"Columns present: {list(raw.columns)[:60]}")
        out[canon] = raw[col]
    for canon, cands in OPTIONAL.items():
        col = _pick(raw, cands)
        if col is not None:
            out[canon] = raw[col]

    out["row_order"] = np.arange(len(out))            # file order = event order in a game
    out["gid"] = out["gid"].str.strip()
    out["inning"] = pd.to_numeric(out["inning"]).astype(int)
    out["bat_home"] = _to_flag(out["bat_home"])
    out["outs_pre"] = pd.to_numeric(out["outs_pre"]).astype(int)
    out["runs"] = pd.to_numeric(out["runs"], errors="coerce").fillna(0).astype(int)
    for b in ("br1_pre", "br2_pre", "br3_pre"):
        v = out[b].astype(str).str.strip()
        out[b[:3]] = (~v.isin(["", "0", "nan", "None"])).astype(int)
        out = out.drop(columns=b)
    out["is_pa"] = _to_flag(out["is_pa"]) if "is_pa" in out else 1
    if "event" in out:                                # drop Retrosheet "no play" rows
        out = out[out["event"].astype(str).str.strip().str.upper() != "NP"]

    # Game id layout: HHHYYYYMMDDN (home team, date, game number 0/1/2)
    out["home_team"] = out["gid"].str[:3]
    out["game_num"] = out["gid"].str[-1].astype(int)
    out["date"] = pd.to_datetime(out["gid"].str[3:11], format="%Y%m%d")
    out["season"] = year

    if "bat_team" not in out or "fld_team" not in out:
        if "vis_team" not in out:
            raise KeyError(f"{plays_path}: need batting/fielding team columns or a visiting-team column.")
        out["bat_team"] = np.where(out["bat_home"] == 1, out["home_team"], out["vis_team"])
        out["fld_team"] = np.where(out["bat_home"] == 1, out["vis_team"], out["home_team"])

    # gameinfo.csv (optional): game type and scheduled innings (7-inning doubleheaders).
    gtype = out.get("gametype")
    gi_path = _find_file(season_dir, "gameinfo")
    if gi_path is not None:
        gi = pd.read_csv(gi_path, dtype=str, keep_default_na=False)
        gcol = _pick(gi, REQUIRED["gid"])
        gi_ids = gi[gcol].str.strip()
        tcol, icol = _pick(gi, OPTIONAL["gametype"]), _pick(gi, ["innings"])
        if gtype is None and tcol:
            gtype = out["gid"].map(dict(zip(gi_ids, gi[tcol])))
        if icol:
            out["sched_innings"] = pd.to_numeric(out["gid"].map(dict(zip(gi_ids, gi[icol]))),
                                                 errors="coerce")
        manifest.append({"season": year, "file": str(gi_path), "sha256": sha256(gi_path),
                         "rows": len(gi)})
    if regular_season_only:
        if gtype is not None:
            keep = gtype.astype(str).str.lower().str.contains("regular|^r$", regex=True)
            out = out[keep.values]
        else:
            warnings.warn(f"{year}: no game-type information; assuming the files hold "
                          "regular-season games only.")

    keep_cols = ["gid", "season", "date", "game_num", "home_team", "row_order", "inning",
                 "bat_home", "bat_team", "fld_team", "pitcher", "outs_pre", "br1", "br2",
                 "br3", "runs", "is_pa"] + [c for c in ("batter", "sched_innings") if c in out]
    return out[keep_cols].reset_index(drop=True), manifest


def load_seasons(years, raw_dir, regular_season_only=True):
    frames, manifest = [], []
    for y in years:
        df, m = load_season(y, raw_dir, regular_season_only)
        frames.append(df)
        manifest += m
    plays = pd.concat(frames, ignore_index=True)
    plays = plays.sort_values(["gid", "row_order"], kind="stable").reset_index(drop=True)
    return plays, manifest
