"""End-to-end and leakage tests on synthetic data.

Run with `pytest -q` or, without pytest, `python tests/test_pipeline.py`.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bullpen.availability import unavailable_reasons  # noqa: E402
from bullpen.build import run  # noqa: E402
from bullpen.config import Config  # noqa: E402
from bullpen.synth import write_synthetic  # noqa: E402

SEASONS = dict(warmup_seasons=(2014,), train_seasons=(2015, 2016, 2017),
               val_seasons=(2018,), test_seasons=(2019,))
_TMP = Path(tempfile.mkdtemp(prefix="bullpen_test_"))
_CACHE = {}


def _read(path):
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, parse_dates=["date"])


def build(tag="full", seasons=None, **over):
    if tag in _CACHE:
        return _CACHE[tag]
    raw = _TMP / "raw"
    if not raw.exists():
        write_synthetic(raw, range(2014, 2020))
    cfg = Config(raw_dir=str(raw), out_dir=str(_TMP / tag), **{**SEASONS, **(seasons or {}), **over})
    m = run(cfg)
    res = (cfg, m, {k: _read(v) for k, v in m["outputs"].items()})
    _CACHE[tag] = res
    return res


def test_outputs_and_splits():
    cfg, m, t = build()
    assert set(t["states"]["split"]) == {"train", "validation", "test"}
    assert m["decision_funnel"]["eligible_decisions"] > 50
    re = pd.read_csv(Path(cfg.out_dir) / "re_matrix.csv", index_col=0)["expected_runs"]
    assert len(re) == 24
    assert re["0_000"] < re["0_111"] and re["2_000"] < re["0_000"]   # more runners / fewer outs -> more runs


def test_decision_filters():
    cfg, _, t = build()
    d = t["decisions"]
    assert d["entry_inning"].between(7, 9).all()
    assert d["entry_margin"].between(cfg.margin_min, cfg.margin_max).all()
    assert (d["n_available"] >= 2).all()
    assert (d["best_rating"] <= d["alt_rating"]).all()
    assert (d["best_pitcher"] != d["alt_pitcher"]).all()


def test_best_is_available_and_unused():
    _, _, t = build()
    c, d = t["candidates"], t["decisions"]
    m = d.merge(c, left_on=["decision_id", "best_pitcher"], right_on=["decision_id", "pitcher"])
    assert len(m) == len(d) and (m["available"] == 1).all()
    assert not m["unavail_reasons"].fillna("").str.contains("used_in_game").any()


def test_rest_rule():
    cfg = Config()
    base = dict(pitched_d1=0, pitched_d2=0, bf_d1=0, apps_prev4=0, pitched_earlier_today=0)
    assert unavailable_reasons(base, False, cfg) == []
    assert "back_to_back" in unavailable_reasons({**base, "pitched_d1": 1, "pitched_d2": 1}, False, cfg)
    assert "heavy_prev_day" in unavailable_reasons({**base, "pitched_d1": 1, "bf_d1": 10}, False, cfg)
    assert "workload_4d" in unavailable_reasons({**base, "apps_prev4": 3}, False, cfg)
    assert "used_in_game" in unavailable_reasons(base, True, cfg)


def test_no_lookahead_in_ratings():
    """Ratings in the test season must not change when later test-season games are deleted."""
    cfg, _, t = build()
    raw_full = Path(cfg.raw_dir)
    cut = pd.Timestamp("2019-04-25")
    raw_cut = _TMP / "raw_cut"
    shutil.copytree(raw_full, raw_cut)
    p = pd.read_csv(raw_cut / "2019" / "plays.csv", dtype=str)
    p[pd.to_datetime(p["gid"].str[3:11]) <= cut].to_csv(raw_cut / "2019" / "plays.csv", index=False)
    cfg2 = Config(raw_dir=str(raw_cut), out_dir=str(_TMP / "cut"), **SEASONS)
    m2 = run(cfg2)
    a = t["candidates"]
    b = _read(m2["outputs"]["candidates"])
    key = ["decision_id", "pitcher"]
    a = a[(a["season"] == 2019) & (a["date"] <= cut)].set_index(key)["rating"]
    b = b[(b["season"] == 2019)].set_index(key)["rating"]
    common = a.index.intersection(b.index)
    assert len(common) > 20
    assert np.allclose(a.loc[common], b.loc[common])


def test_manifest_records_inputs():
    cfg, m, _ = build()
    saved = json.loads((Path(cfg.out_dir) / "manifest.json").read_text())
    assert len(saved["input_files"]) == 6 and all(len(f["sha256"]) == 64 for f in saved["input_files"])


def test_model_ladder_runs_and_skips_test_rows():
    from bullpen.model_y1 import POST_EVENT, RAW_INPUTS, load, run as run_model
    cfg, _, _ = build()
    states, _ = load(cfg)
    assert set(states["split"]) == {"train", "validation"}
    assert not POST_EVENT & RAW_INPUTS
    r = run_model(cfg, boot=20, out_dir=str(_TMP / "y1"))
    m = r["results"]["validation_all_relief"]["models"]
    assert set(m) == {"M0", "M1", "M2"} and all(np.isfinite(v["log_loss"]) for v in m.values())


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1; print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if fails else 0)
