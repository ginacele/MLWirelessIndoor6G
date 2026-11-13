## `graphs.py`

Draws `residual_distance.png`, `variograms.png`, `qq_plots.png` and
`dual_slope.png`. Three reasons it is parked:

1. **Its inputs do not exist.** It reads `results/binned_residuals.csv`,
   `results/variograms.csv`, `results/residuals.csv` and
   `results/spec_results_full_precision.csv`. No script in `scripts/` writes any
   of the four, so the producer is missing.
2. **Different directory convention.** It expects a `results/` directory beside
   itself; every script in `scripts/` reads and writes `data/`.
3. **It assumes analyses that are not in the repo.** The spec labels
   `POOLED_anchored` and `POOLED_building_offsets`, the variograms, and the
   `Comms_dual_*` fixed-breakpoint sweep have no producer here, and the
   dual-slope analysis does not appear in either results document.
