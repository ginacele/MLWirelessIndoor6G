# data/

This directory ships empty. It holds both the raw inputs and every derived CSV.

## Raw inputs — put these here first

Six files, which `pool_data.py` globs as `Prx_*_classified.csv`:

```
Prx_Comms_C1_classified.csv     Prx_Comms_C2_classified.csv
Prx_Library_C1_classified.csv   Prx_Library_C2_classified.csv
Prx_SSE_C1_classified.csv       Prx_SSE_C2_classified.csv
```

They are the published measurements of Perdomo-Reyes, Galvan-Tejada and
Meneses-Viveros, *Path Loss Dataset from Field Measurements at 3.5 GHz for the
Fifth Generation of Wireless Communications in Indoor Environments*, Scientific
Data 13:521 (2026), [doi:10.1038/s41597-026-06650-4](https://doi.org/10.1038/s41597-026-06650-4),
with data on OSF. They are not our measurements, which is why they are not
vendored in this repo by default — see the provenance note in the root README
before deciding to commit them.

Schema notes `pool_data.py` relies on: a BOM on the leading `Coord.` header
(hence `utf-8-sig`), units in the headers (`Distance (m)`, `P_rx (dBm)`), an
`Elevator` column in Library only, and two stray unnamed columns in SSE C2.

## Derived outputs

Written here by the scripts, in the order given in the root README:

```
pooled.csv  pooled_all.csv            pool_data.py
primary_results.csv                   fit_models.py
a_vs_e.csv                            a_vs_e.py
baseline_results.csv                  baseline_fits.py
transfer_results.csv                  transfer.py
scenario_metrics.csv                  scenarios.py
scenario_coefficients.csv             scenarios.py
scenario_residuals.csv                scenarios.py
ml_results.csv                        ml_models.py
local_mean_results.csv                local_mean.py
coverage_results.csv                  coverage.py
qfunction_results.csv                 qfunction.py
sse_los_nlos.csv                      sse_los_nlos.py
```

`offset_diagnosis.py` writes nothing; its output is the printed tables.

## scenario_metrics.csv columns

Documented here because the scripts no longer carry comments. One row per
`(spec, scenario, role, condition, k, repeat)` — every k-shot row is a single
draw, not a summary, so medians, means, quantiles and error bands are all
computable from it and nothing is pre-aggregated.

| Column | Meaning |
|---|---|
| `spec` | `A` or `E` |
| `scenario` | `S1`, `S2` or `S3` |
| `train` | training building |
| `role` | `validate` or `test` |
| `building` | the held-out building |
| `condition` | `blind` or `kshot_intercept` |
| `k` | calibration points requested (0 when blind) |
| `repeat` | draw index (0 when blind, else 0…N_REPEATS-1) |
| `n_test` | points scored |
| `n_cal` | calibration points actually placed — spacing can yield fewer than `k` |
| `mse` | mean squared error, dB² |
| `rmse` | root mean squared error, dB |
| `bias` | mean residual, dB |
| `sd` | residual standard deviation, dB (`sd² + bias² = mse`) |
| `mae` | mean absolute error, dB |
| `p90` | 90th percentile of \|residual\|, dB — the fade-margin tail |

Three things to know when plotting it, all of which used to be comments in
`scenarios.py`: aggregate in `mse` and then take the square root, because
averaging `rmse` across folds is not meaningful; keep `bias` and `sd` as
separate series, because blind transfer failure is mostly bias and intercept
calibration removes bias while leaving `sd` almost unchanged; and plot against
`n_cal` rather than `k`, or stop the x axis where spacing starts to bind, since
beyond that point a larger `k` is the same experiment. `REFERENCE_LEVELS` in
`scenarios.py` holds the constants to draw as horizontal lines.
