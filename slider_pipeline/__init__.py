"""Automated phase-1 slider comparison pipeline.

One sweep = one FIXED caption pair, one frozen base recipe, N variants that
differ in exactly the knobs the sweep varies (loss formulation, alpha, ...).
Every random source is pinned in the spec and recorded in the outputs, so two
variants differ only by the variant — the paired design is what makes recipe
effects measurable at all: unpaired, the between-seed spread of a ladder delta
(sd ~1.0 ln-rms on the trip-hop pair) swamps every recipe effect ever measured
(<= 0.25 ln); paired on the same seeds the same effect is sign-consistent with
sd ~0.18.

Stages (independently runnable, resumable, loud on failure):
  train   -> models/pipeline/<sweep>/<variant>/
  render  -> eval/listen/pipeline/<sweep>/<variant>-seed<k>/   (+ null-seed<k>)
  score   -> eval/listen/pipeline/<sweep>/scores.json          (gates + ranking)
  report  -> eval/listen/pipeline/<sweep>/REPORT.md, ab_metric.csv, index pages

The measurement spec (intended features, bands, audibility constants) is frozen
in the sweep YAML before any search runs, per SCORING.md. The scorer never
reads captions from the optimizer; prompts_file is part of the frozen pair.
"""
