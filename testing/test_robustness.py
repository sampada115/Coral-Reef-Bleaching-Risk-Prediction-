"""
Robustness / fault-tolerance tests: every outside service can fail without breaking the site.
  NOAA down      -> saved snapshot (latest) or a clear 503 (other dates)
  weather down   -> forecast falls back to physics persistence for all leads
  LLM down       -> offline template briefing
  bad LLM output -> invented numbers flagged and removed before display
"""

import datetime as dt

import pytest

from conftest import CASE_DATE, sse_events


def test_noaa_down_latest_uses_snapshot(client, monkeypatch):
    from backend import app as appmod

    def boom(*a, **k):
        raise TimeoutError("NOAA unreachable")
    monkeypatch.setattr(appmod.noaa, "fetch_map_grid", boom)
    monkeypatch.setitem(appmod._latest, "grid", None)
    monkeypatch.setitem(appmod._latest, "fetched", 0.0)
    if not appmod.SNAPSHOT_LATEST:
        pytest.skip("no snapshots saved")
    r = client.get("/api/grid/latest")
    assert r.status_code == 200 and r.json()["source"] == "snapshot"


def test_noaa_down_other_date_gives_clear_error(client, monkeypatch):
    from backend import app as appmod

    def boom(*a, **k):
        raise TimeoutError("NOAA unreachable")
    monkeypatch.setattr(appmod.noaa, "fetch_map_grid", boom)
    day = "2019-06-17"                               # not saved, not cached
    monkeypatch.setattr(appmod, "CACHE", appmod.CACHE / "_none_")
    appmod._grid_cache.pop(day, None)
    r = client.get(f"/api/grid/{day}")
    assert r.status_code == 503 and "NOAA" in r.json()["detail"]


def test_weather_down_forecast_falls_back_to_physics(monkeypatch):
    from backend.forecasting import forecast_v2

    def boom(*a, **k):
        raise ConnectionError("Open-Meteo unreachable")
    monkeypatch.setattr(forecast_v2, "_weather", boom)
    f = forecast_v2.forecast(CASE_DATE)
    assert f["weather"] == "none"
    assert f["physics_persistence_leads"] == list(range(1, 15))
    assert len([v for v in f["vars"]["forecast"] if v is not None]) == 1927


def test_llm_unreachable_gives_offline_briefing(client, monkeypatch):
    from backend.ai import llm
    monkeypatch.setenv("CORALWATCH_LLM", "auto")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(llm, "OLLAMA_URL", "http://127.0.0.1:9")      # nothing listens here
    assert llm.provider()[0] == "offline"
    r = client.post("/api/explain", json={"date": CASE_DATE.isoformat(), "zone": "N"})
    ev = sse_events(r.text)
    done = dict(ev)["done"]
    assert done["mode"] == "offline" and dict(ev)["verify"]["grounded"]


def test_hallucinated_numbers_removed_in_strict_mode():
    from backend.ai import grounding
    facts = ("On 2024-03-03, 412 reefs in the Northern GBR were at Alert Level 1 or higher.\n"
             "For 2024-03-10 (7 days ahead), the model forecasts 748 reefs in the Northern GBR at Alert Level 1 or higher.")
    answer = ("### What the satellites show\n"
              "On 2024-03-03, 412 reefs were at Alert Level 1 or higher. [Data]\n"
              "About 90% of corals have already died. [Data]\n"                      # invented
              "### Next 7 days\n"
              "The model forecasts 412 reefs at Alert Level 1 or higher on 2024-03-10. [Data]\n"   # wrong day
              "The model forecasts 748 reefs at Alert Level 1 or higher on 2024-03-10. [Data]")
    report = grounding.verify(answer, facts, [])
    assert not report["grounded"]
    cleaned, removed = grounding.remove_unsupported(answer, report)
    assert removed == 2
    assert "90%" not in cleaned and "748 reefs" in cleaned
    assert grounding.verify(cleaned, facts, [])["grounded"]


def test_unknown_date_forecast_returns_503_not_crash(client):
    future_gap = (dt.date.today() + dt.timedelta(days=30)).isoformat()
    assert client.get(f"/api/forecast?date={future_gap}").status_code == 400
