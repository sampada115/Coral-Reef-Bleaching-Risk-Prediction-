"""
Model and data validation tests (research-specific testing).
  * data integrity: dataset shape, Marine Park mask, NOAA rules reproduced on the data
  * experimental hygiene: season splits disjoint, model choice on validation only
  * no leakage: changing FUTURE ocean data must not change today's forecast or embedding
  * reproducibility: the same input gives the same output
  * regression guard: the saved results still beat the baselines
Needs forecast/data/cube.npz and weather.npz (git-ignored; skipped if missing).
"""

import datetime as dt
import json

import numpy as np
import pytest

from conftest import CASE_DATE, ROOT

DATA = ROOT / "forecast" / "data"
MODELS = ROOT / "forecast" / "6_saved_models" / "v2"
RESULTS = ROOT / "forecast" / "5_results" / "v2_max7_final"
needs_data = pytest.mark.skipif(not (DATA / "cube.npz").exists() or not (DATA / "weather.npz").exists(),
                                reason="forecast/data not present")


@pytest.fixture(scope="module")
def cube():
    from dataset import Cube
    return Cube()


@pytest.fixture(scope="module")
def drivers(cube):
    from paths import WEATHER
    from physnet import Drivers
    return Drivers(WEATHER, cube.lats, cube.lons)


# ── Data integrity ────────────────────────────────────────────────────
@needs_data
def test_dataset_shape_and_mask(cube):
    assert cube.x.shape[1:] == (4, 107, 106)
    assert int(cube.valid.sum()) == 1927
    assert cube.dates[0] == dt.date(2018, 1, 1) and cube.dates[-1] == dt.date(2025, 12, 31)
    assert len(cube.dates) == len(set(cube.dates)), "no duplicate days"
    assert all(b > a for a, b in zip(cube.dates, cube.dates[1:])), "days in order"


@needs_data
def test_values_physically_plausible(cube):
    v = cube.valid
    ssta, hs, dhw, sst = (cube.x[:, i][:, v] for i in range(4))
    assert np.nanmin(sst) > 15 and np.nanmax(sst) < 35          # degC, tropical sea
    assert np.nanmin(hs) >= 0 - 1e-3 and np.nanmax(hs) < 6       # HotSpot is never negative
    assert np.nanmin(dhw) >= 0 - 1e-3 and np.nanmax(dhw) < 25
    assert np.nanmin(ssta) > -6 and np.nanmax(ssta) < 6


@needs_data
def test_noaa_dhw_formula_holds_on_data(cube):
    """DHW(t) = sum of HotSpot >= 1 over the 84 days ending on t, / 7 (NOAA CRW v3.1)."""
    v = cube.valid
    t = cube.index[CASE_DATE]
    window = [cube.index[CASE_DATE - dt.timedelta(days=k)] for k in range(84)]
    hs = cube.x[window, 1][:, v]
    dhw_calc = np.nansum(np.where(hs >= 1, hs, 0), 0) / 7
    dhw_noaa = cube.x[t, 2][v]
    ok = ~np.isnan(dhw_noaa)
    assert np.corrcoef(dhw_calc[ok], dhw_noaa[ok])[0, 1] > 0.99


# ── Experimental hygiene ──────────────────────────────────────────────
def test_season_splits_are_disjoint():
    from dataset import SPLITS
    tr, va, te = (set(SPLITS[k]) for k in ("train", "val", "test"))
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert max(tr) < min(va) < min(te), "train before validation before test (no future in training)"


@needs_data
def test_samples_respect_splits(cube):
    from dataset import SPLITS, season_of
    for split in ("train", "val", "test"):
        for t in cube.samples(split)[::25]:
            target = cube.dates[t] + dt.timedelta(days=7)
            assert season_of(target) in SPLITS[split]


def test_hybrid_and_model_choice_used_validation_only():
    hybrid = json.loads((RESULTS / "hybrid.json").read_text())
    assert hybrid["selected_on"] == "validation seasons"
    res = json.loads((RESULTS / "results.json").read_text())
    assert res["validation_scores"], "model choice recorded from validation scores"
    best = max(res["validation_scores"], key=res["validation_scores"].get)
    # the best set-up on validation is selected, then its 3 training runs are averaged
    assert res["selected"].split(" - ensemble")[0] == best.rsplit(" (seed", 1)[0]


# ── No leakage ────────────────────────────────────────────────────────
def _scramble_future(cube, day, days=14, seed=0):
    t = cube.index[day]
    rng = np.random.default_rng(seed)
    fut = slice(t + 1, t + 1 + days)
    cube.x[fut] = (rng.normal(0, 3, cube.x[fut].shape) + 30).astype(cube.x.dtype)
    from coralwatch import geo
    cube.baa = geo.baa_from(cube.x[:, 1], cube.x[:, 2]).astype(np.float32)
    if hasattr(cube, "_baa7"):
        del cube._baa7
    return t


@needs_data
def test_forecast_unchanged_when_future_ocean_data_changes(cube, drivers):
    """If the model could see the answer, scrambling the next 14 days would change its output."""
    from dataset import Cube
    from physnet import PhysNet
    day = CASE_DATE
    base = PhysNet.load(MODELS / "max7_s0.pt", cube, drivers).predict_all([cube.index[day]])[0]
    altered = Cube()
    t = _scramble_future(altered, day)
    after = PhysNet.load(MODELS / "max7_s0.pt", altered, drivers).predict_all([t])[0]
    assert np.allclose(base.astype(np.float32), after.astype(np.float32), atol=1e-3)


@needs_data
def test_embedding_unchanged_when_future_changes(cube, drivers):
    import importlib.util
    from dataset import Cube
    from physnet import PhysNet
    spec = importlib.util.spec_from_file_location("emb", ROOT / "forecast" / "7_embeddings" / "extract_embeddings.py")
    emb = importlib.util.module_from_spec(spec); spec.loader.exec_module(emb)
    day = CASE_DATE
    e1 = emb.embed_batch(PhysNet.load(MODELS / "max7_s0.pt", cube, drivers), [cube.index[day]])
    altered = Cube()
    t = _scramble_future(altered, day)
    e2 = emb.embed_batch(PhysNet.load(MODELS / "max7_s0.pt", altered, drivers), [t])
    assert np.allclose(e1, e2, atol=1e-4)


# ── Reproducibility ───────────────────────────────────────────────────
@needs_data
def test_prediction_is_deterministic(cube, drivers):
    from physnet import PhysNet
    m = PhysNet.load(MODELS / "max7_s0.pt", cube, drivers)
    t = cube.index[CASE_DATE]
    a, _ = m.predict_all([t]); b, _ = m.predict_all([t])
    assert np.array_equal(a, b)
    p = a[0].astype(np.float32).sum(1)                       # class probabilities sum to 1
    assert np.allclose(p[:, cube.valid], 1, atol=1e-2)


# ── Regression guard on the reported results ──────────────────────────
def test_results_beat_baselines():
    res = json.loads((RESULTS / "results.json").read_text())
    sel = res["selected"]
    for lead in ("7", "14"):
        ours = res["leads"][lead][sel]["accuracy"]
        pers = res["leads"][lead]["Persistence"]["accuracy"]
        assert ours[1] > pers[0], f"lead {lead}: lower CI bound must beat persistence"
    assert res["leads"]["7"][sel]["accuracy"][0] >= 0.82
    sig = res["significance_vs_persistence_7d"][sel]
    assert max(sig.values()) < 0.05
    hybrid = json.loads((RESULTS / "hybrid.json").read_text())
    for lead, h in hybrid["leads"].items():
        assert h["test"]["accuracy"][0] >= h["test_persistence"]["accuracy"][0], lead


def test_real_forecast_check_beats_persistence():
    r = json.loads((RESULTS / "real_forecast_check.json").read_text())["results"]
    real = r["Forecast-weather model + real archived forecasts"]["accuracy"][0]
    assert real > r["Persistence"]["accuracy"][0] + 0.10
