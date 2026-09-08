window.BENCHMARK_DATA = {
  "lastUpdate": 1788847533660,
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
      },
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
          "id": "3d8b6d5afb44e17196ec4a81f0daf7fa50a7f3f4",
          "message": "fix(ci): dependency-check cleanups\n\n- npm ci in frontend so lockfile actually gets scanned\n- --disableNodeAudit (NPM Audit API unreachable from runners, wasted 60s)\n- drop stray empty root package-lock.json\n- note: NVD refresh needs an NVD API key secret; runs with --noupdate until then\n\nCo-Authored-By: Claude <noreply@anthropic.com>",
          "timestamp": "2026-09-04T13:12:00+05:30",
          "tree_id": "514ddcf8de8e47c48733a17d3b550b60e0e31a27",
          "url": "https://github.com/Atharva0177/Solar-Energy-Prediction/commit/3d8b6d5afb44e17196ec4a81f0daf7fa50a7f3f4"
        },
        "date": 1788507834567,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_benchmark_xgboost_single_step",
            "value": 456.2572293385182,
            "unit": "iter/sec",
            "range": "stddev: 0.00013971895594544186",
            "extra": "mean: 2.1917460934258512 msec\nrounds: 289"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_api_forecast_xgboost",
            "value": 15.077319971450594,
            "unit": "iter/sec",
            "range": "stddev: 0.0009767841344162308",
            "extra": "mean: 66.3247846363633 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_api_forecast_lstm",
            "value": 27.68085292950609,
            "unit": "iter/sec",
            "range": "stddev: 0.0008927833923025234",
            "extra": "mean: 36.12605444444457 msec\nrounds: 27"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_api_history",
            "value": 102.55387728060909,
            "unit": "iter/sec",
            "range": "stddev: 0.00023386676022256123",
            "extra": "mean: 9.75097213793086 msec\nrounds: 87"
          }
        ]
      },
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
          "id": "cb2ed32df53fdd626233ba9afaa03e1f1d63acf4",
          "message": "docs: verify PRD section 56 acceptance criteria against repo artifacts\n\n29/30 done with evidence pointers; Random Forest marked skipped\n(never implemented in this build, already documented in configs and\nComparison page).\n\nCo-Authored-By: Claude <noreply@anthropic.com>",
          "timestamp": "2026-09-04T13:45:17+05:30",
          "tree_id": "1e66747cde8c2df5472ed4211007b509fc34f0a9",
          "url": "https://github.com/Atharva0177/Solar-Energy-Prediction/commit/cb2ed32df53fdd626233ba9afaa03e1f1d63acf4"
        },
        "date": 1788509824847,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_benchmark_xgboost_single_step",
            "value": 515.4850647211954,
            "unit": "iter/sec",
            "range": "stddev: 0.00014196097526826277",
            "extra": "mean: 1.9399204136803823 msec\nrounds: 307"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_api_forecast_xgboost",
            "value": 17.299552111566857,
            "unit": "iter/sec",
            "range": "stddev: 0.001073741889192891",
            "extra": "mean: 57.80496475000518 msec\nrounds: 12"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_api_forecast_lstm",
            "value": 32.26863599886313,
            "unit": "iter/sec",
            "range": "stddev: 0.002980309358926404",
            "extra": "mean: 30.989844133332173 msec\nrounds: 30"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_api_history",
            "value": 108.05373266130844,
            "unit": "iter/sec",
            "range": "stddev: 0.0003861304781412243",
            "extra": "mean: 9.254654840425305 msec\nrounds: 94"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mandavkaratharva@gmail.com",
            "name": "Atharva_177",
            "username": "Atharva0177"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "01775678b0ffbc69d76e7d50a6e399e3996ae7cf",
          "message": "Create SECURITY.md",
          "timestamp": "2026-09-05T12:27:54+05:30",
          "tree_id": "454e9d8e1ff314fff9f3c30fa0b8f3ee865a5550",
          "url": "https://github.com/Atharva0177/Solar-Energy-Prediction/commit/01775678b0ffbc69d76e7d50a6e399e3996ae7cf"
        },
        "date": 1788591573838,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_benchmark_xgboost_single_step",
            "value": 679.5051754519797,
            "unit": "iter/sec",
            "range": "stddev: 0.00010426210706218986",
            "extra": "mean: 1.4716591368635858 msec\nrounds: 453"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_api_forecast_xgboost",
            "value": 22.88311908743134,
            "unit": "iter/sec",
            "range": "stddev: 0.0006009501433118343",
            "extra": "mean: 43.70033631251147 msec\nrounds: 16"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_api_forecast_lstm",
            "value": 41.66233747799637,
            "unit": "iter/sec",
            "range": "stddev: 0.0007262158247880582",
            "extra": "mean: 24.00249387178869 msec\nrounds: 39"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_api_history",
            "value": 136.96092567396352,
            "unit": "iter/sec",
            "range": "stddev: 0.00017683646846082508",
            "extra": "mean: 7.301352521379034 msec\nrounds: 117"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Atharva_177",
            "username": "Atharva0177",
            "email": "mandavkaratharva@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "01775678b0ffbc69d76e7d50a6e399e3996ae7cf",
          "message": "Create SECURITY.md",
          "timestamp": "2026-09-05T06:57:54Z",
          "url": "https://github.com/Atharva0177/Solar-Energy-Prediction/commit/01775678b0ffbc69d76e7d50a6e399e3996ae7cf"
        },
        "date": 1788662124115,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_benchmark_xgboost_single_step",
            "value": 396.0568570130257,
            "unit": "iter/sec",
            "range": "stddev: 0.0009632390718137951",
            "extra": "mean: 2.524890005798111 msec\nrounds: 345"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_api_forecast_xgboost",
            "value": 15.996211957406699,
            "unit": "iter/sec",
            "range": "stddev: 0.0008637215255475579",
            "extra": "mean: 62.514800545448615 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_api_forecast_lstm",
            "value": 29.863013513978174,
            "unit": "iter/sec",
            "range": "stddev: 0.002550446407397073",
            "extra": "mean: 33.48623873916554 msec\nrounds: 23"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_api_history",
            "value": 107.18524146013061,
            "unit": "iter/sec",
            "range": "stddev: 0.00016297239471186538",
            "extra": "mean: 9.329642648348813 msec\nrounds: 91"
          }
        ]
      },
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
          "id": "b292b39c6b31ccd5b04a9d5c331b7663bf1f756b",
          "message": "Merge branch 'main' of https://github.com/Atharva0177/Solar-Energy-Prediction",
          "timestamp": "2026-09-08T11:32:54+05:30",
          "tree_id": "483fa2a21140184a58bdcbb7e994fef3481b8a53",
          "url": "https://github.com/Atharva0177/Solar-Energy-Prediction/commit/b292b39c6b31ccd5b04a9d5c331b7663bf1f756b"
        },
        "date": 1788847492969,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_benchmark_xgboost_single_step",
            "value": 520.9733616733713,
            "unit": "iter/sec",
            "range": "stddev: 0.00010595812287858573",
            "extra": "mean: 1.9194839382727569 msec\nrounds: 324"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_api_forecast_xgboost",
            "value": 17.679507922367428,
            "unit": "iter/sec",
            "range": "stddev: 0.0006738790904703781",
            "extra": "mean: 56.562660249997045 msec\nrounds: 12"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_api_forecast_lstm",
            "value": 32.02753458805832,
            "unit": "iter/sec",
            "range": "stddev: 0.0008966800438656282",
            "extra": "mean: 31.22313387096791 msec\nrounds: 31"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_api_history",
            "value": 107.93980433172365,
            "unit": "iter/sec",
            "range": "stddev: 0.0003033982024309215",
            "extra": "mean: 9.264422945651928 msec\nrounds: 92"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mandavkaratharva@gmail.com",
            "name": "Atharva_177",
            "username": "Atharva0177"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "613686763fc35a486be92bec5b662bc674979eef",
          "message": "Create LICENSE",
          "timestamp": "2026-09-08T11:33:51+05:30",
          "tree_id": "8e30d772624eb2a7a4e564ed4cada396d62547b6",
          "url": "https://github.com/Atharva0177/Solar-Energy-Prediction/commit/613686763fc35a486be92bec5b662bc674979eef"
        },
        "date": 1788847532963,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_benchmark_xgboost_single_step",
            "value": 517.0427563039595,
            "unit": "iter/sec",
            "range": "stddev: 0.00010791921774382836",
            "extra": "mean: 1.9340760271905233 msec\nrounds: 331"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_api_forecast_xgboost",
            "value": 17.670718694503538,
            "unit": "iter/sec",
            "range": "stddev: 0.0010091114511244369",
            "extra": "mean: 56.590793916664474 msec\nrounds: 12"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_api_forecast_lstm",
            "value": 32.941783469521646,
            "unit": "iter/sec",
            "range": "stddev: 0.0006448796399061254",
            "extra": "mean: 30.35658348386689 msec\nrounds: 31"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_api_history",
            "value": 106.50973449798451,
            "unit": "iter/sec",
            "range": "stddev: 0.00014909573162275383",
            "extra": "mean: 9.388813188891415 msec\nrounds: 90"
          }
        ]
      }
    ]
  }
}