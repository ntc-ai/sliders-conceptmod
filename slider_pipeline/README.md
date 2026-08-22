# Slider comparison pipeline

Operational runbook for `scripts/slider_pipeline.py` and the
`slider_pipeline/` package. Measurement contract: [SCORING.md](../SCORING.md).
Campaign notes and retracted metrics: [MUSIC3.md](../MUSIC3.md).

**Scope: transformer sliders only.** The gates (especially G4 song identity)
were frozen on transformer attacks. A working LM half changes the arrangement
by design, so same-seed onset/envelope correlation against scale 0 is ~0 and
the instrument vetoes shipped, ears-approved LM checkpoints. Do not train,
render, score, or rank LM halves here.

## Intent

One sweep = one **frozen caption pair** + one **frozen base recipe** + N
variants that differ only in the knobs they name. Every seed is pinned in the
YAML so ranking is a **paired** comparison: unpaired, between-seed sd of a +1
rms delta is ~1.0 ln on the trip-hop pair, which swamps every recipe effect
ever measured (≤ 0.25 ln).

The pipeline exists so a silent argparse default cannot poison a comparison
unseen (that already cost a week). Finished work is adopted only after a
config-echo check against the sidecar.

## Stages

| stage | GPU? | writes | notes |
|---|---|---|---|
| `selftest` | no | stdout | Synthetic collapse / silence / hiss / song-replace / no-op / reversed / dead-pole ladders must all veto. Run first. |
| `onboard` | yes | `eval/listen/pipeline/onboard-<pair>/` | G0: certify the pair from **condition-interpolation** ladders (`scripts/render_cond_interp.py`), not caption swaps. Exit 3 if rejected. |
| `train` | yes | `models/pipeline/<sweep>/<variant>/` | Calls `train_lora_music3.py`. Resumable: existing `*_last.json` is adopted if it echoes the spec. |
| `render` | yes | `eval/listen/pipeline/<sweep>/<variant>-seed<k>/` plus `null-seed<k>/` | Paired ladders + zero-only null seeds for G6. Scores **`_last`**, never `_best` (the probe does not predict renders). |
| `score` | no | `scores.json` | Gates are vetoes; the scalar ranks gate-passers only. |
| `report` | no | `REPORT.md`, `ab_metric.csv` | Human-facing. Also tries `scripts/build_listen_index.py`. |
| `confirm` | yes | `confirm-<winner>.json` | Holdout seeds never used for ranking + one long-duration render. |
| `score-folders` | no | optional JSON/CSV | Legacy acceptance path over already-rendered `<root>/<variant>-seed<k>/` trees. |

```bash
PY=/home/mikkel/anaconda3/envs/minimax-music3/bin/python
$PY scripts/slider_pipeline.py selftest
# new pair — certify BEFORE training:
$PY scripts/slider_pipeline.py onboard \
  --prompts_file conceptmod/textsliders/data/prompts-cand-dust-v1.yaml \
  --cache_dir cache/dust-v1 \
  --out_root eval/listen/pipeline/onboard-dust \
  --intended centroid:+1
$PY scripts/slider_pipeline.py train   slider_pipeline/specs/phase1-loss-triphop.yaml --gpu 0
$PY scripts/slider_pipeline.py render  slider_pipeline/specs/phase1-loss-triphop.yaml --gpu 0
$PY scripts/slider_pipeline.py score   slider_pipeline/specs/phase1-loss-triphop.yaml
$PY scripts/slider_pipeline.py report  slider_pipeline/specs/phase1-loss-triphop.yaml
$PY scripts/slider_pipeline.py confirm slider_pipeline/specs/phase1-loss-triphop.yaml --winner NAME --gpu 0
```

`--only V0-nmse,L-mse` shards `train` / `render`. Null-seed renders run only on
the invocation that owns the baseline, so two `--only` shards cannot race.

## Writing a spec

Copy `slider_pipeline/specs/phase1-loss-triphop.yaml`. `load_spec` refuses
anything that would make the comparison unpaired or unscoreable:

- `cache_dir` must already exist. The pipeline never runs the AR stage
  (`skip_ar` is required in practice; build the cache with one manual
  `train_lora_music3.py` pass first).
- `base` must pin `seed`, `cond_seeds`, and `eval_seed`.
- `prompts_file`, `cache_dir`, `save_dir`, `name`, and `device` are
  **pipeline-owned** — do not put them in `base` or variant `args`.
- `scales` must include `0.0` and both signs.
- `over_scale` must lie beyond the ladder (G7).
- `compare` needs ≥ 3 seeds; `holdout` must not overlap `compare`.
- `baseline` must name a variant. Variant `role` is one of
  `candidate | baseline | floor | alpha | control`.
- `axis.intended` maps scored features to `±1`. Known features:
  `rms_pc`, `centroid`, `hi4k`, `flatness`, `crest`, `width_db`,
  `flux_p90_med`. Loudness axes set `level_axis: true` (wider G2 band).

YAML parses a bare `null:` key as Python `None`. The loader accepts `null`,
`null_seeds`, or the None-key spelling for the G6 seed pool.

## GPU and the hardcoded interpreter

`slider_pipeline/stages.py` pins
`PY=/home/mikkel/anaconda3/envs/minimax-music3/bin/python`. `--gpu N` sets
`CUDA_VISIBLE_DEVICES=N` and always passes `--device 0` to the trainer and
renderer (index 0 of that single visible GPU).

Renders use `--raw_scales --accept_silent --retry_seeds 0` so a silent pole
stays on the requested seed (a retry would unpair the comparison).

## What the gates actually do

Implemented in `slider_pipeline/gates.py`; numbers are frozen there and in
SCORING.md. Any failure rejects — no trade-offs.

| gate | vetoes |
|---|---|
| G0 (onboard) | No certifiable acoustic direction (sign-flip permutation on cond-interp spans). Prompt problem, not a training problem. |
| G1 | Near-silence vs the same-seed zero clip. |
| G2 | Level explosion / collapse (median / worst dB bands; wider if `level_axis`). |
| G3 | Reversed or dead pole (Spearman + per-side endpoint sign). |
| G4 | Song replacement (onset / envelope / chroma vs same-seed zero, ≥ 0.9). |
| G5 | Hiss / roughness / crest / quiet-whine; `must_not_move` extras. A dust/crackle axis that *is* texture should list `flatness` as intended so G5 does not punish the concept. |
| G6 | Score inside the 95th percentile of zero-slider seed-noise (`null` seeds). |
| G7 | Silence at `over_scale`. Long duration + holdout seeds are `confirm`, not ranking. |

`floor` variants set the retrain-noise band: a candidate whose paired effect
sits inside that band is a **TIE**, not a win.

## Worked example and known outcome

`specs/phase1-loss-triphop.yaml` varies loss / gain / alpha / `x0_per_row` on
the trip-hop ↔ glossy pair. Report:
`eval/listen/pipeline/phase1-loss-triphop/REPORT.md`.

Every variant was vetoed. The archive “gain_penalty 2.0 recovers +0.25 ln rms”
cell did not reproduce under pinned seeds (instrument drift: legacy ladders
used `_best`, the pipeline scores `_last`). That pair is a ceiling for
*ranking* losses; it is not a passing baseline.

The one transformer checkpoint that cleared all six gates on a render is
**`dust-tf-v1`** (sidecar `models/dust-tf-v1/dust-tf-v1_last.json`, promoted
from overnight `F-dust-nopen`, score 0.203 over 10 seeds). The trainer
defaults were then changed to that recipe — see MUSIC3.md “Current
transformer trainer defaults”. A better-posed next sweep is loss variants
**on the dust pair**, which has a passing baseline to beat.

## Troubleshooting

| symptom | cause |
|---|---|
| spec load: `cache_dir missing` | Build the condition cache first; the pipeline will not encode AR implicitly. |
| spec load: `unpinned seeds` / `pipeline-owned` key | Move `seed`/`cond_seeds`/`eval_seed` into `base`; remove `name`/`device`/`save_dir` from args. |
| `config echo mismatch` / refuse to adopt | Sidecar disagrees with the spec (silent argparse default, or a leftover run). Delete that variant dir or fix the spec — do not ignore the echo. |
| `expected exactly one *_last.json` | Extra or missing sidecars in the variant save dir. |
| onboard / cond-interp exits on “chunked denoise” | Duration > 8 s (one denoise window). Stay at 4 s (default) or 8 s. |
| LM half fails G3/G4/G6 | Expected. Use `probe_lm_axis_signal.py`, train metrics, `render_end_ab.py`, and ears. |
| `--device 1` on the trainer while the pipeline used `--gpu 1` | Pipeline already isolated GPU 1 as `cuda:0`. Leave `--device` to the pipeline. |
