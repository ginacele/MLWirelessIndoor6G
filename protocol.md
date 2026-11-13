# Analysis protocol

> **DRAFT — assembled, not authored.** Every rule below was lifted from a
> docstring or comment that was in `scripts/` before those files were stripped
> to numeric output. Nothing here is new policy and nothing was inferred, but it
> needs your review before being cited as the `protocol.md` the results document
> refers to.
>
> **This file is now the only record of the rationale.** The scripts no longer
> carry it. The `→` pointers name the file that *implements* each rule, so you
> can still find the code, but you will not find the explanation there — only
> here. The three rules at the end were never in the code; they come from the
> results documents and are marked as such.

The results document states that analysis conventions, and the specific error
each exists to prevent, are recorded here. The pattern throughout is that each
rule was adopted *after* a specific mistake, so the error is recorded with it.

---

## Data handling

**1. Screen for physical impossibility, and drop rather than correct.**
A reading at or above the transmit power means more power arrived than was sent.
The one such row (Comms C2, coordinate C-36, `P_rx = +70 dBm`) is dropped, not
repaired. *Error prevented:* the intended value is almost certainly `-70 dBm`,
but it cannot be verified, and substituting it would be fabricating data.
→ `pool_data.py`

**2. Coordinates and the published label never enter a fit.**
`Coord.` is retained for spatial blocking only. The `LOS/NLOS Classification`
column is copied to `audit_label` and used only for auditing. *Error prevented:*
the published label turns out to be exactly the zero-obstruction indicator, so
using it as a feature would be feeding the model one of its own inputs twice.
→ `pool_data.py`

**3. No building dummies anywhere.**
The `building` column is for splitting folds, not for fitting. *Error
prevented:* a building dummy makes the model unable to transfer, which is the
quantity the whole project is trying to measure.
→ `pool_data.py`

**4. Flag the noise floor, don't drop it.**
Rows within 10 dB of the stated -115 dBm coverage threshold are marked
`near_noise_floor` and kept. *Error prevented:* dropping them removes the
longest-distance points and so removes the lever arm on the slope, which is a
distance-range truncation masquerading as a censoring test.
→ `pool_data.py`, and the explicit caution in `offset_diagnosis.py` test 4

**5. Declare what comes from outside the fit.**
Only two quantities are not fitted: the 33.3 dB anchor and the 5 m threshold.
Published reference values (3GPP TR 38.901, ITU-R P.1238) are used for
comparison and are never fed into a fit.
→ `a_vs_e.py`, section "WHERE EACH NUMBER COMES FROM"

**6. The anchor is 33.3 dB, not 43.3 dB.**
The dataset defines path loss as `P_tx - P_rx` with no antenna-gain correction,
so the quantity is system loss and the 1 m anchor is 43.3 less the two nominal
5 dBi gains. *Error prevented:* a 43.3 dB anchor biases the exponent down by
roughly 1.0 and lands it spuriously near the 3GPP InH NLOS value of 3.19. That
agreement would be an artifact.
→ `baseline_fits.py`, "REFERENCE VALUES" warning

---

## Inference

**7. Never report a difference between subgroups from binned medians alone.**
Run a formal test first. *Error prevented:* in the configuration analysis the
binned medians looked like a clean log-linear gradient and were in fact a
threshold at 5 m.
→ `fit_models.py`, "Protocol rule, enforced below" and `formal_subgroup_test`

**8. Interval overlap, not point-estimate spread.**
*Error prevented:* the pre-registered "exponent spread < 0.5" criterion compared
point estimates with no reference to their uncertainty. It passes under spec A
and fails under spec E while the honest inference is identical either way. It
was the wrong test and has been retired.
→ `fit_models.py`, closing note

**9. Block bootstrap the exponent; do not quote OLS standard errors.**
Resample grid-letter blocks within configuration, never rows. *Error
prevented:* at 1 m grid pitch against roughly 2 m shadowing decorrelation,
neighbouring points are near-duplicates, so a row bootstrap gives intervals up
to 3.1x too narrow. *Caveat recorded with the rule:* blocked resampling is
standard in spatial statistics, but no propagation paper was found that
block-bootstraps the path loss exponent specifically. Defensible by analogy;
not established practice in this field.
→ `fit_models.py` and `baseline_fits.py`, both `block_bootstrap_n`

**10. Seed a bootstrap from its own data, not from a shared generator.**
*Error prevented:* with one shared RNG the confidence interval depends on how
many earlier calls were made, which showed up as a single subset reporting two
different intervals within one run.
→ `offset_diagnosis.py`, `boot_median`

**11. Drop degenerate features explicitly.**
Features with no variance in a subset are unidentifiable and are reported as
dropped rather than silently fitted. Per-material obstruction counts have zero
variance in some buildings.
→ `baseline_fits.py`, `drop_degenerate`

**12. Pre-register predictions before running.**
Each experiment records what it expects, in the file, before the run.
→ `transfer.py` (P1-P3), `ml_models.py` (M1-M4), and the pair-mean prediction in
`local_mean.py`

**13. A handful of matched pairs is not evidence until independence is checked.**
*Error prevented:* two apparently independent distance-matched comparisons in
SSE turned out to be the same two coordinates measured under two antenna
heights, and the effect they appeared to show sat well inside the ordinary
spread of the C2-C1 difference.
→ `sse_los_nlos.py`, sections 1-3

---

## Validation

**14. Spatial blocks, never random rows.**
Within-building cross-validation holds out whole grid-letter bands. *Error
prevented:* at this grid pitch a randomly held-out point almost always has a
near-duplicate in training, so a random split measures interpolation rather than
prediction.
→ `ml_models.py`, `run_within`; same rationale in `local_mean.py`, `letter_folds`

**15. Compare against a baseline fitted on the same data.**
*Error prevented:* beating an uncalibrated textbook model proves nothing, and
that comparison is the standard critique of this literature.
→ `ml_models.py`, docstring

**16. Calibration points must be at least 3 m apart, and curves plotted against
points actually placed.**
*Error prevented:* unspaced points are correlated neighbours whose effective
sample size is far below k, so the calibration curve flatters itself. Spacing on
distance-from-transmitter alone instead of physical position caps the achievable
count at roughly 8 / 7 / 4 points, which makes every k >= 10 the same experiment
and produces a flat tail that means nothing. Check `n_cal` against `k` before
reading any curve.
→ `transfer.py` and `scenarios.py`, both `spaced_idx`; `SPACING_MODE` note

**17. Averaged targets: build windows from training rows only, and recompute the
floor for each target.**
*Error prevented:* a window mean that includes test points leaks test values
into their own target. Separately, averaging shrinks variance mechanically, so
an averaged target compared against the raw target's floor will look better than
it is — and overlapping windows collapse the ceiling faster than they reduce
genuine error (SSE returns a 1.31 dB ceiling at a 4 m radius in a 164 m²
building, which is clearly an artifact).
→ `local_mean.py`, `window_mean` / `blocked_rmse_window` and the caveats block

**18. Aggregate in MSE, then take the square root.**
Averaging MSE across folds is meaningful; averaging RMSE is not.
→ `scenarios.py`, column notes

**19. Report bias and SD separately, never RMSE alone.**
*Error prevented:* blind transfer failure is mostly bias, and intercept
calibration removes bias while leaving SD almost untouched. An RMSE-only curve
hides that entirely — RMSE falls dramatically while point-to-point scatter is
unchanged.
→ `scenarios.py`, column notes

---

## Sourced to the results documents rather than to code

These three are stated as conclusions in the results write-ups and are applied
in the analysis, but no script records them as a rule. Confirm the wording.

**20. Do not calibrate a model that is already unbiased.** Below roughly 2-3 dB
of blind bias, estimating a shift from a handful of points adds variance and
makes the prediction worse. SSE → Comms degrades from +0.030 to +0.048 under
calibration.

**21. Refit the intercept only, not the slope, at small k.** Refitting both
overfits: 8.58 dB against 7.61 dB with Comms as the test building.

**22. In-sample fit quality does not predict transferability, and must not be
used to choose a training site.** Library has the lowest residual sigma at
5.97 dB and transfers worst in both directions.

---

## Appendix: pre-registered predictions

Rule 12 requires predictions to be written before a run. These were, and they
were recorded in the script docstrings; the strip to numeric output removed them
from the code, so they are preserved verbatim here. The scripts still print the
checks against them — `transfer.py` prints the bias spread and the k-shot
progression, `ml_models.py` prints sections labelled M1 to M4.

### `transfer.py`

**P1** Blind bias lands in 4–9 dB, matching the fitted offset spread (+20.75
Comms, +15.83 Library, +12.09 SSE under spec E).

**P2** `kshot_intercept` recovers most of the blind penalty by k ≈ 10, since the
intercept standard error goes as σ/√k and σ ≈ 6.6 dB.

**P3** The obstruction coefficient does *not* transfer between Comms and
Library. `L_total` is +2.87 in Comms and +0.48 in Library, because Library's
dominant drywall count carries a non-physical coefficient. So Comms→Library and
Library→Comms stay the worst pairs even after intercept correction, which k-shot
cannot fix.

Reference points cited with them: within-building sigma 6.58 dB (n-weighted
rms), shared-offset pooled sigma 7.74 dB — blind transfer cannot beat the
latter.

### `ml_models.py`

**M1** PURE beats PARAM within-building by 1–2 dB, and loses on transfer.

**M2** HYBRID ≥ PARAM everywhere, because its distance law is structural rather
than learned.

**M3** SSE→Comms is where PURE fails hardest. Trees cannot extrapolate past
their training support, and SSE stops at 15.9 m while Comms reaches 30.1 m.
PARAM extrapolates by construction.

**M4** Any ML advantage concentrates in Library and SSE, the two buildings with
significant residual curvature (p = 0.0043, 0.0060).

Reference points cited with them: within-building sigma 6.58 dB, shared-offset
pooled sigma 7.74 dB, best transfer so far 6.57 dB (Comms→SSE at k=20).

### `local_mean.py`

The variance decomposition was predicted before the averaged fits were run: if
RMSE² = model² + fading², subtracting the independently measured fading from the
raw error gives the model's own error, and predicts what the error becomes on a
two-point average. Predicted 6.24 / 4.81 / 4.83 dB against observed 6.28 / 4.86
/ 5.24 dB for Comms / Library / SSE.
