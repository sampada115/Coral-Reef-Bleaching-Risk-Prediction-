# backend/: web server, live forecast and AI

```
backend/
├── app.py              the web server: serves the website + API (run: ./run.sh)
├── forecasting/
│   ├── forecast_v2.py  live forecast with the final model (NOAA data + weather forecast + ENSO/MJO)
│   ├── forecast_v1.py  first corrected model, used only if v2 is missing
│   └── forecast_now.py terminal tool: run the real-time forecast yourself
├── ai/
│   ├── build_corpus.py turns the reef report PDFs into 3,180 searchable passages
│   ├── rag.py          finds the most relevant passages (only reports published before the date)
│   └── llm.py          builds the prompt, streams Claude's answer, offline fallback
└── data/
    ├── corpus.json     the report passages
    └── cache/          cached NOAA days, weather and map data
```

## API

| Endpoint | What it returns |
|---|---|
| `GET /api/health` | What's working: live NOAA data, model, LLM, corpus |
| `GET /api/grid/{date}` | NOAA map data for a day (`latest` = today, refreshed every 3 h) |
| `GET /api/forecast?date=` | 1–14 day forecast issued on a date (default: latest NOAA day) |
| `POST /api/explain` | Streamed AI briefing or answer for a date and zone |

## Real-time data (automatic)

Every 3 hours:
1. the newest NOAA Coral Reef Watch days (112 days of history, cached),
2. Open-Meteo weather (past 40 days + next 16 days of forecast),
3. ENSO/MJO,
4. the forecast is re-run.

If the weather service is down, forecasts fall back to physics persistence.

```bash
.venv/bin/python -m backend.forecasting.forecast_now
.venv/bin/python -m backend.forecasting.forecast_now --date 2024-03-03 --save forecast.json
.venv/bin/python -m backend.ai.build_corpus        # rebuild report passages after adding PDFs
```
