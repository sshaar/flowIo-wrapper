# Training plan: learning from compensation editing sessions

Status: draft v1, 2026-10-07. Owner: nb634@cornell.edu.

## 0. What we are and are not trying to learn

The spillover matrix is, in principle, computable from clean single-stain controls; the built-in algorithms do
that well. What scientists add by hand is a mix of (a) corrections for things the algorithm cannot see
(degraded tandem dyes, bead-vs-cell autofluorescence mismatch, a wrong negative population, drift between
controls and samples) and (b) subjective tweaks that make a plot "look right" but are not more correct.
A model that imitates the edits learns both. The plan below therefore:

- starts with analysis and a **flagging** model (which cells will a scientist touch, and which way), which
  is useful even if the exact values are noisy;
- treats **regression of the correction** as a second-stage goal gated on data volume and on having plot
  histograms;
- measures everything against **inter-operator agreement**, which is the ceiling for any imitation model;
- keeps an **objective spillover score** alongside the human labels so we can tell (a) from (b).

## 1. Data: what the collector produces

From `flowio-capture build-dataset` (see README for details):

| file | unit | role |
|---|---|---|
| `accepted.jsonl` | one per (session, matrix) | inputs (context, controls), baseline, accepted, per-cell `status` |
| `cell_directions.jsonl` | one per judgment edit | direction labels, view, `plot_refs` |
| `preference_pairs.jsonl` | one per (accepted, earlier state) | ranking signal; `revised_export` strongest |
| `cell_hard_negatives.jsonl` | one per rejected value | undone / reverted / overshoot |
| `tolerances.jsonl` | one per edited cell | how far off is still acceptable |
| `plots.jsonl` | one per logged plot | what the scientist saw (64×64 histogram, matrix hash) |

Key fields to respect:
- `clean_baseline`: train the delta only where True (baseline was an auto/acquisition matrix).
- Cell `status`: `unexamined` cells are **not** negatives for the flagging model; drop or down-weight them.
- `replay_errors` and `summary.skipped`: audit before every training run.
- Dataset build parameters (`eps`, `drag_dt`) are part of the label definition: record them with each build.

## 2. Phase 0: readiness (before any modeling)

Gate to pass before Phase 1:

1. Plugin wired and validated on the real app (`docs/INSTALL.md` §4).
2. Privacy sign-off; decide whether `capture_plots` is on. **Without plots, Phases 3–4 are weak.**
3. Pilot: 2+ scientists, 2+ weeks, ≥ 20 labeled sessions. Audit `summary.json`: skip rate, desync notes,
   unclean baselines, cell-status mix, median edits per session.
4. Fix instrumentation gaps found in the pilot (e.g. missing `source` on loaded matrices, missing view events)
   before scaling collection. Label quality is decided here, not in the model.

## 3. Phase 1: analysis and baselines (no ML, ~2 weeks after pilot)

Purpose: understand the process, define the metrics, and set baselines a model must beat.

Analyses (notebooks in `analysis/`, to be added):
- Edit frequency and magnitude per (fluorochrome, detector) pair, per instrument, per panel, per operator.
- Fraction of sessions where the auto matrix is accepted untouched (the majority, expected).
- Distribution of `delta` for edited cells; direction bias (do people systematically under-compensate?).
- Tolerance brackets per pair; time per edit; drag vs discrete edits.
- Inter-operator study: same controls, different scientists. This is the agreement ceiling. If none exist
  naturally, arrange a small one (5 experiments × 3 scientists).
- Objective score: for each accepted matrix, compute the median-alignment residual of positive vs negative
  control populations on each spillover axis from `controls_at_accept` (the quantity algorithms optimize).
  Compare human-accepted vs auto. This tells us how often (a) vs (b) is happening.

Baselines to beat later:
- B0: auto matrix as-is (delta = 0).
- B1: per-pair mean correction (fluor, detector, instrument) from training sessions.
- B2: algorithmic recompute from control stats (if the stats are rich enough).

Metrics (defined here, frozen for later phases):
- Flagging: per-cell AUROC / AUPRC for "will be edited"; direction accuracy on edited cells.
- Correction: per-cell L1 in spillover units; **within-tolerance rate** (|pred − accepted| ≤ cell bracket,
  default bracket 0.005 when no observed tolerance); whole-matrix max error.
- Always reported against the inter-operator ceiling.

## 4. Phase 2: flagging model ("check these cells")

Target: given the auto matrix and context, predict per cell P(edited) and direction. Realistic at ~100–200
labeled sessions; the first thing worth putting in front of a scientist.

- **Rows**: one per off-diagonal cell of each accepted matrix with status ∈ {edited, examined_accepted}.
  Exclude `unexamined` (unknown label). Label = status == edited; direction from `cell_directions`
  (sign of accepted − baseline).
- **Features** (tabular first):
  - fluorochrome, detector, instrument model, laser/filter (categorical);
  - auto value, row/col sums, rank of the cell's spillover within its row;
  - control stats for the source fluorochrome: positive/negative medians and rSDs on the source and target
    detectors, positive-to-negative separation, brightness; control type (beads/cells);
  - panel size; whether the pair was viewed (`viewed`) — for analysis only, **not** as a model input (leaks
    the outcome at inference time).
- **Model**: gradient-boosted trees (LightGBM) with grouped CV. Then a small per-matrix transformer over
  cells with (fluor, detector) embeddings, to use cross-cell context (compensation cells are coupled).
- **Validation**: grouped by session (all states of a session in one fold), and separately by panel and by
  operator to measure generalization. Calibration curves — the output is a hint, so calibration matters.
- **Deliverable**: a ranked "cells to check" list per matrix, evaluated by recall@3 against what the scientist
  actually edited.

## 5. Phase 3: correction regression and preference learning

Target: predict `accepted − baseline` per cell. Gate: ≥ 500 clean-baseline sessions across ≥ 3 operators
and ≥ 2 instruments, or multi-lab pooling. Below that, expect regression to the per-pair mean (B1).

- Loss: Huber on delta, weighted by 1/tolerance where a bracket exists.
- Preference pairs: a margin ranking loss on a matrix-level scorer (accepted ≻ rejected), weighting
  `revised_export` pairs highest and `hard` pairs above the rest. Mid-drag states are already excluded.
- Cell hard negatives: hinge terms pushing predictions away from `undone`/`reverted`/`overshoot` values.
- Architecture: the Phase-2 cell transformer with a regression head; shared embeddings so panels of different
  sizes pool.
- Report within-tolerance rate vs B0/B1/B2 and vs inter-operator agreement. If the model is not clearly
  better than B1 within tolerance, stop here and ship Phase 2.

## 6. Phase 4: plot-conditioned edit policy (the real model)

Target: imitate the scientist's *decision from what they saw*. Input: histogram at edit time (sample, and
control if present), current cell value, context. Output: the correction (or next edit). This is where the
plots matter; without them this phase does not exist.

- Data: `cell_directions` rows joined to `plots` via `plot_refs`; each is (state, observation) → action.
- Model: small CNN over the 64×64 histogram (log-count, two channels: sample/control) + tabular features →
  delta. Sequential variant: predict the next edit given the trajectory so far (behavior cloning).
- Also derive a *computed* feature from the histogram that mirrors what humans look at: median of the
  positive population vs negative population along the target axis. If that feature alone matches the model,
  the model has learned the standard rule, which is useful to know.
- Evaluation as in Phase 3, plus a held-out operator.

## 7. Phase 5: validation beyond imitation

- Compare model, human, and algorithm on the objective score from Phase 1. Where the model agrees with
  humans *against* the objective score, inspect those cases: they are either real expert knowledge or
  shared bias, and the plots will usually show which.
- Prospective study: show Phase-2 hints to scientists and measure whether edit time or revised-export rate
  changes. Note this ends the "purely passive" property and needs its own consent; design it as a separate
  mode, off by default.

## 8. Engineering plan

- `training/` package (to add): loaders that read the JSONL files into pandas; a `splits.py` that produces
  grouped folds by session / panel / operator / time and asserts no session leaks across folds (intermediate
  states, pairs, and plots of one session must share a fold).
- `analysis/` notebooks for Phase 1; results committed as Markdown with figures.
- Dataset versioning: every build writes `summary.json`; add `build_params` (eps, drag_dt, wrapper version,
  git SHA) and a content hash; name datasets by that hash.
- Compute: everything through Phase 3 is laptop-scale. Phase 4 runs on a single GPU.
- Dependencies for training stay out of the collector package (the collector must remain dependency-free).

## 9. Risks and mitigations

| risk | mitigation |
|---|---|
| Too little data; one panel, one operator | pool across labs with a shared salt policy; Phase 2 before Phase 3; report ceilings |
| No cell-level events from the SDK | polling `on_matrix_changed`; accept coarser trajectories |
| No plot access | Phase 4 dropped; Phases 1–3 still stand |
| Export is a noisy acceptance signal | `revised_export` handling; audit skip reasons; consider a second acceptance signal (workspace save after export) |
| Model learns human bias | objective score in Phase 1/5; never deploy Phase 3+ without it |
| Leakage across session states in CV | grouped splits enforced in `splits.py` with an assertion |

## 10. Milestones

| # | milestone | gate |
|---|---|---|
| M0 | plugin on real app, pilot data | ≥ 20 labeled sessions, audit clean |
| M1 | Phase-1 analysis report | metrics frozen, ceiling measured |
| M2 | flagging model | beats B0 on recall@3 by a margin that matters to scientists |
| M3 | regression model | within-tolerance rate > B1 and approaching ceiling |
| M4 | plot-conditioned model | beats M3 on held-out operator |
| M5 | prospective study | consent + separate mode |
