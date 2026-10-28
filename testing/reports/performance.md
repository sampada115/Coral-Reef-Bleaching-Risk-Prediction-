# Performance test

Machine: 8 cores, 8.0 GB RAM, Python 3.14.4

| Operation | Median | 95th percentile |
|---|---|---|
| Server start-up | 1.5 s |  |
| Health check | 0.7 ms | 1.4 ms |
| Map data (saved day) | 1.2 ms | 1.8 ms |
| Forecast (first call: load 3 models + run) | 7.5 s |  |
| Forecast (cached) | 268.7 ms | 309.9 ms |
| AI briefing, offline mode | 40.6 ms | 126.0 ms |

| Concurrent users | Requests | Median | 95th pct | Throughput | Errors |
|---|---|---|---|---|---|
| 10 | 40 | 21.5 ms | 37.4 ms | 382.3 req/s | 0 |
| 25 | 100 | 50.2 ms | 73.3 ms | 454.5 req/s | 0 |
| 50 | 200 | 89.2 ms | 207.1 ms | 407.6 req/s | 0 |

Server memory: 148 MB idle, 444 MB with the forecast models loaded.
