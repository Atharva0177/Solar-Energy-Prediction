window.BENCHMARK_DATA = {
  "lastUpdate": 1788504914277,
  "repoUrl": "https://github.com/Atharva0177/Solar-Energy-Prediction",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "mandavkaratharva@gmail.com",
            "name": "Atharva0177",
            "username": "Atharva0177"
          },
          "committer": {
            "email": "mandavkaratharva@gmail.com",
            "name": "Atharva0177",
            "username": "Atharva0177"
          },
          "distinct": true,
          "id": "3d2f429f17d94ba9a3d0c68e0dc3b83a78f267b4",
          "message": "fix(ci): grant contents:write to ml-benchmarks for gh-pages push\n\nbenchmark-action committed results but push was denied for\ngithub-actions[bot] (default token permissions).\n\nCo-Authored-By: Claude <noreply@anthropic.com>",
          "timestamp": "2026-09-04T12:23:23+05:30",
          "tree_id": "29391e3d8568a2703024af3a9fce94a820d993fd",
          "url": "https://github.com/Atharva0177/Solar-Energy-Prediction/commit/3d2f429f17d94ba9a3d0c68e0dc3b83a78f267b4"
        },
        "date": 1788504913770,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_benchmark_xgboost_single_step",
            "value": 548.3604672338655,
            "unit": "iter/sec",
            "range": "stddev: 0.00010017074110884687",
            "extra": "mean: 1.8236179661972578 msec\nrounds: 355"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_api_forecast_xgboost",
            "value": 19.371298198932486,
            "unit": "iter/sec",
            "range": "stddev: 0.0011682715904752082",
            "extra": "mean: 51.62276630768649 msec\nrounds: 13"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_api_forecast_lstm",
            "value": 34.19631722513105,
            "unit": "iter/sec",
            "range": "stddev: 0.002752944059557295",
            "extra": "mean: 29.24291506060468 msec\nrounds: 33"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_api_history",
            "value": 116.14760395594008,
            "unit": "iter/sec",
            "range": "stddev: 0.0001920719589511637",
            "extra": "mean: 8.609734216982593 msec\nrounds: 106"
          }
        ]
      }
    ]
  }
}