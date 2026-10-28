# Performance test

Machine: 8 cores, 8.0 GB RAM, Python 3.14.4

| Operation | Median | 95th percentile |
|---|---|---|
| Server start-up | 1.5 s |  |
| Health check | 0.6 ms | 0.7 ms |
| Map data (saved day) | 178.1 ms | 178.8 ms |
| Forecast (first call: load 3 models + run) | 83.7 s |  |
| Forecast (cached) | 263.8 ms | 265.7 ms |
| AI briefing, offline mode | 37.7 ms | 38.9 ms |

| Concurrent users | Requests | Median | 95th pct | Throughput | Errors |
|---|---|---|---|---|---|
| 10 | 40 | 1792.6 ms | 2093.0 ms | 5.4 req/s | 0 |
| 25 | 100 | 4397.6 ms | 4584.0 ms | 5.7 req/s | 0 |
| 50 | 200 | 8914.0 ms | 9241.5 ms | 5.6 req/s | 0 |

Server memory: 149 MB idle, 44 MB after the tests.
