# Indoor path loss at 3.5 GHz — analysis code

Measurement-driven analysis of the Cinvestav 3.5 GHz indoor path-loss campaign
across three buildings (Communications, Library, SSE) and two antenna
configurations, supporting the two results documents in this project.


## Layout

```
requirements.txt
run_all.sh         runs the whole path in dependency order
data/              raw Prx_*_classified.csv inputs, and every derived CSV
scripts/           the reproduction path, in run order below
wip/               code that does not yet run against this repo
```

`data/` is both input and output directory, which is what the scripts assume.
`wip/graphs.py` instead expects a `results/` directory; reconcile that when it
comes back onto the path.

## First run

`data/` ships empty. Put the six raw CSVs in it before running anything — see
`data/README.md` for the filenames and where they come from.

```sh
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# drop the six Prx_*_classified.csv files into data/
./run_all.sh logs/          # or run scripts/pool_data.py first, then the rest
```

`scikit-learn` is needed only by `ml_models.py`; `matplotlib` only by
`wip/graphs.py`. Nothing has been rerun against the real measurements since the
repo was reorganised, so treat the first full run as a check that the published
counts (2,283 unfiltered and 2,226 NLOS) still come out.

## Run order

Run `pool_data.py` first. Everything after it depends only on the pool, not on
each other, so steps 2–12 can run in any order or in parallel.

| # | Script | Reads | Writes |
|---|---|---|---|
| 1 | `pool_data.py` | `Prx_*_classified.csv` | `pooled.csv`, `pooled_all.csv` |
| 2 | `offset_diagnosis.py` | `pooled_all.csv` | — (prints) |
| 3 | `fit_models.py` | `pooled.csv` | `primary_results.csv` |
| 4 | `a_vs_e.py` | `pooled.csv` | `a_vs_e.csv` |
| 5 | `baseline_fits.py` | `pooled.csv` | `baseline_results.csv` |
| 6 | `transfer.py` | `pooled.csv` | `transfer_results.csv` |
| 7 | `scenarios.py` | `pooled.csv` | `scenario_metrics.csv`, `scenario_coefficients.csv`, `scenario_residuals.csv` |
| 8 | `ml_models.py` | `pooled.csv` | `ml_results.csv` |
| 9 | `local_mean.py` | `pooled.csv` | `local_mean_results.csv` |
| 10 | `coverage.py` | `pooled.csv` | `coverage_results.csv` |
| 11 | `qfunction.py` | `pooled.csv` | `qfunction_results.csv` |
| 12 | `sse_los_nlos.py` | `pooled.csv` | `sse_los_nlos.csv` |

```sh
python scripts/pool_data.py
for s in offset_diagnosis fit_models a_vs_e baseline_fits transfer \
         scenarios ml_models local_mean coverage qfunction sse_los_nlos; do
    python "scripts/$s.py"
done
```

## What each script is for

- **`pool_data.py`** — the single shared input. Physical-validity screening
  (one impossible `P_rx = +70 dBm` row dropped), sub-metre points removed,
  noise-floor and reference-level flags added, `N_total` built.
- **`offset_diagnosis.py`** — the ~19 dB system offset, and the two mechanisms
  ruled out for it: analyser reference level and noise-floor censoring. Also
  the short-range LOS delta, the paired configuration test, and the
  50 %-exceedance Q-function evaluation.
- **`fit_models.py`** — primary results table. Specs A and E per building and
  pooled, block-bootstrap intervals, and the interval-overlap test that
  replaces the retired "exponent spread < 0.5" criterion.
- **`a_vs_e.py`** — A against E in isolation. Source of the `t(δ)` and `Δσ`
  columns in the results tables.
- **`baseline_fits.py`** — per-material obstruction coefficients, VIF, anchored
  against floating intercept, and the ALL / NLOS / LOS comparison.
- **`transfer.py`** — nine train/test folds, blind and k-shot, with the
  calibration curve.
- **`scenarios.py`** — the three-scenario rotation (S1/S2/S3), each building
  training once and held out twice. Long-format output, nothing pre-aggregated.
- **`ml_models.py`** — parametric, hybrid and pure gradient boosting on
  identical features under spatial blocks and the same transfer folds.
- **`local_mean.py`** — independent fading estimate from the C1/C2 pairing,
  information ceilings recomputed per averaging target, variance decomposition.
- **`coverage.py`** — coverage at a fixed threshold, calibration bins, fade
  margins, and transfer of the coverage claim.
- **`qfunction.py`** — the same Q-function machinery at one threshold, plus the
  worked-points table. Largely overlaps `coverage.py`; see Open items.
- **`sse_los_nlos.py`** — why two distance-matched SSE pairs cannot support a
  configuration-dependent obstruction effect. Supporting negative result, not
  on the publication path.

## Two pooling modes

`NLOS_ONLY` in `pool_data.py` selects what lands in `pooled.csv`.

- **`False` (default)** writes the unfiltered frame. This is the input behind
  the headline numbers: exponents 2.535 / 2.443 / 2.607 under spec A, and the
  scenario and transfer folds with train n = 1,384 / 687 / 212.
- **`True`** writes the NLOS frame instead, which is the input behind the
  NLOS-only refit where SSE moves from 2.607 to 3.215 under spec A.

`pooled_all.csv` is always unfiltered in either mode, because
`offset_diagnosis.py` needs the zero-obstruction points.

`a_vs_e.py`, `baseline_fits.py` and `sse_los_nlos.py` do their own LOS/NLOS
subsetting and so require the unfiltered pool. Under `NLOS_ONLY = True` their
subsets go degenerate rather than raising, so each warns when its LOS rows are
missing.

## Data provenance

The six `Prx_*_classified.csv` inputs are the published measurements of
Perdomo-Reyes, Galvan-Tejada and Meneses-Viveros, *Path Loss Dataset from Field
Measurements at 3.5 GHz for the Fifth Generation of Wireless Communications in
Indoor Environments*, Scientific Data 13:521 (2026),
[doi:10.1038/s41597-026-06650-4](https://doi.org/10.1038/s41597-026-06650-4).

## Starting the repository

```sh
git init
git add .
git commit -m "Initial commit: 3.5 GHz indoor path loss analysis pipeline"
git branch -M main
git remote add origin <your remote>
git push -u origin main
```
