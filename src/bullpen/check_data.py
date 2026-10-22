"""Check that raw Retrosheet files are in place and readable before building.

    python -m bullpen.check_data [--config configs/default.json]

Expected layout (one folder per season):
    data/raw/retrosheet/2014/plays.csv      (+ optional gameinfo.csv)
    ...
    data/raw/retrosheet/2025/plays.csv
"""
import argparse
import sys
import warnings

from .config import Config
from .io_retrosheet import load_season


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    cfg = Config.from_json(ap.parse_args(argv).config)

    ok = True
    for y in cfg.all_seasons:
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                df, _ = load_season(y, cfg.raw_dir, cfg.regular_season_only)
            note = f"  ({w[0].message})" if w else ""
            print(f"{y}: OK  {df['gid'].nunique():>5} games  {len(df):>7} events{note}")
            if df["gid"].nunique() < 2000:
                print(f"      warning: {y} looks incomplete (a full season is ~2,430 games; 2020 is ~900)")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"{y}: MISSING/INVALID  {e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
