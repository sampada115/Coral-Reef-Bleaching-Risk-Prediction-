"""
Integration tests: the modules working together through the real API.
NOAA data -> forecast model -> zone statistics -> facts -> report search -> briefing -> website.
Uses the cached 2024-03-03 data, so it runs without internet (except ENSO/MJO, which
fall back to physics persistence if unreachable - the test accepts both).
"""

import re

import pytest

from conftest import CASE_DATE, ROOT, sse_events


@pytest.fixture(scope="module")
def forecast(client):
    r = client.get(f"/api/forecast?date={CASE_DATE}")
    assert r.status_code == 200, r.text
    return r.json()


def test_health_reports_all_components(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok"
    assert h["forecast_model"], "a trained forecast model must be loaded"
    assert h["corpus_passages"] > 3000
    assert h["llm"]["provider"] in ("local", "claude", "offline")


def test_forecast_structure_and_value_ranges(forecast):
    f = forecast
    assert f["issue_date"] == CASE_DATE.isoformat() and f["target"] == "max7"
    assert f["leads"] == list(range(1, 15))
    n = len(f["lats"]) * len(f["lons"])
    for k in map(str, f["leads"]):
        levels = [v for v in f["by_lead_levels"][k] if v is not None]
        probs = [v for v in f["by_lead_p_alert"][k] if v is not None]
        assert len(f["by_lead_levels"][k]) == n
        assert len(levels) == 1927, "one forecast per Marine Park cell"
        assert set(levels) <= {0, 1, 2, 3, 4}
        assert all(0 <= p <= 1 for p in probs)


def test_forecast_hybrid_uses_physics_for_short_leads(forecast):
    if forecast["weather"] == "none":
        assert forecast["physics_persistence_leads"] == list(range(1, 15))
    else:
        assert forecast["physics_persistence_leads"] == [1, 2, 3, 4]


def test_zone_counts_are_consistent(forecast):
    for k, zones in forecast["forecast_zones_by_lead"].items():
        for z in zones:
            assert 0 <= z["reefs_at_alert"] <= z["reefs_total"], (k, z)
            assert 0 <= z["mean_alert_probability"] <= 1


def test_grid_to_forecast_to_briefing(client, offline_llm, forecast):
    r = client.post("/api/explain", json={"date": CASE_DATE.isoformat(), "zone": "N"})
    assert r.status_code == 200
    ev = sse_events(r.text)
    kinds = [e for e, _ in ev]
    assert kinds[0] == "context" and kinds[-1] == "done" and "verify" in kinds
    context = ev[0][1]
    facts = context["facts"]
    # The briefing's facts must carry the same forecast the /api/forecast endpoint returned
    n_api = next(z for z in forecast["forecast_zones"] if z["zone"] == "N")["reefs_at_alert"]
    assert facts["forecast"]["zone"]["reefs_at_alert"] == n_api
    # No hindsight: every source was published before the briefing date's cutoff year (2023 for Mar 2024)
    years = [s["published"] for s in context["sources"] if s.get("published")]
    assert all(int(str(y)[:4]) <= 2023 for y in years), years
    verify = dict(ev)["verify"]
    assert verify["grounded"] and not verify["unverified_numbers"]


def test_question_answer_flow(client, offline_llm):
    r = client.post("/api/explain", json={"date": CASE_DATE.isoformat(), "zone": "C",
                                          "question": "Which areas are most at risk next week?"})
    assert r.status_code == 200
    assert [e for e, _ in sse_events(r.text)][-1] == "done"


def test_website_assets_load(client):
    html = client.get("/").text
    assert "CoralWatch" in html
    for ref in re.findall(r'(?:src|href)="([^"#:]+\.(?:js|css))"', html):
        assert client.get("/" + ref.lstrip("./")).status_code == 200, ref
    assert (ROOT / "website" / "data" / "gbr.js").exists()
