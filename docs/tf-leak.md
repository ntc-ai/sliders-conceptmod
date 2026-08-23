# Is the default Music 3 TF leak real?

**ok** — do not change `--loss nmse --target_mode axis` and do not add
`--attributes` to transformer sliders.

The CPU 2-D suite's `needs_help` (leak ~0.33) is the unused **male/female**
axis. Shipped TF yamls never write that attribute. The real second axis in
the catalog is **BPM / genre sitting inside the energy (and distortion)
captions**, and the default loss is doing what it is told: it fits
`pos − neg`. `pole`, `nmse_ortho`, and `gain_penalty` do not take BPM out
of the teacher. `--attributes` is the averaging MUSIC3.md already measured
as destroying TF style sliders.

## Strongest evidence

Energy vs tempo bag-of-words cosine on `pos − neg` is **0.000** — they
share no distinctive adjectives — but the shipped energy pair is
`BPM: 168` vs `BPM: 52` (Δ 116). Tempo is `180` vs `48` (Δ 132). The 2-D
gender leak and this caption are different facts.

On a CPU field whose *x* is the energy yaml's loudness words and whose
*y* is numeric BPM + tempo words, live `music3_slider_loss`:

| pair / method | energy cos | tempo leak |
|---|---:|---:|
| energy, `nmse`/`axis` (default) | 0.91 | **0.47** |
| energy, `pole` | 0.92 | 0.41 |
| energy, `nmse_ortho` | 0.92 | 0.43 |
| energy, `gain_penalty=1` | 0.91 | 0.46 |
| energy + gender `--attributes` | 0.91 | 0.47 |
| **cand-energy (BPM pinned 110)** | **1.00** | **0.00** |
| tempo (control) | 0.00 | (pure tempo) |

Same default loss. The leak appears when the yaml moves BPM; it vanishes
on the fixed-BPM pair that is already in the repo
(`prompts-cand-energy-v1.yaml`).

## Hypotheses

| claim | verdict |
|---|---|
| Leak is only the synthetic male/female prefix; Music 3 TF yamls have no such attribute and are fine | **Accepted for the 0.33 number.** energy / tempo / distortion / space / dust TF rows have `has_attributes: false`. Gender prefixes do not change BPM, so they cannot fix this. |
| Energy / loudness / live / distortion captions are entangled, so a default energy slider will also move grit / live | **Partial.** Energy × grit adjective overlap is **empty** (bow cos 0). Energy × distortion shares one word, `aggressive` (bow cos 0.064). Energy × tempo is the BPM numbers, not shared words. Live v3 pins `BPM: 112` and `Medium energy` on every pole (ΔBPM 0). Distortion TF *renders* do move loudness (below). |
| Multi-row axes that are not parallel already wash the slider; leak is the wrong diagnosis | **Right reason not to add `--attributes`.** MUSIC3.md: multi-row averaging drops TF `cos_axis` to 0.05. Shipped TF sliders are **single-row**. The 0.33 is a single-row gender leak in the toy, not a wash. |
| `pole` / `nmse_ortho` / gain already address common-mode leak | **Discarded for this leak.** Those terms touch the even / `vel_neu`-parallel part. BPM lives in `pos − neg` (the odd teacher). Measured leak stays 0.41–0.47. |

## Caption geometry (no model)

Distinctive-token cosine of `pos − neg` (boilerplate `genre` / `bpm` /
`song` stripped):

|  | energy | tempo | distortion | space | dust | grit | cand-energy |
|---|---:|---:|---:|---:|---:|---:|---:|
| energy | 1 | **0.000** | 0.064 | 0.092 | 0.000 | **0.000** | 0.161 |
| tempo | | 1 | −0.035 | 0 | 0 | 0 | 0 |
| distortion | | | 1 | 0.091 | 0 | 0.125 | 0.079 |
| dust | | | | | 1 | 0 | −0.053 |

BPM deltas: energy **+116**, tempo **+132**, distortion **+52**,
triphop-single **−44**, rhyme-v4 **+38**. Pinned at 0: dust, grit,
cand-energy, live.

Shared adjectives that actually exist: energy ∩ distortion =
`aggressive`; energy ∩ grit = none; energy ∩ cand-energy =
`loud`/`dense`/`quiet`/`sparse` (same intended axis).

## Existing render numbers (quoted from git)

From the LISTEN.md tables and the dust sidecar — single-seed, so treat
as corroboration, not a new campaign:

| source | −2 / 0 / +2 rms (or sidecar %) | reading |
|---|---|---|
| `eval/listen/energy-20s/` | 0.0891 / 0.1119 / 0.1492 | loudness moves; that is the axis |
| `eval/listen/tempo-20s/` | 0.0540 / 0.0544 / 0.0538 | **flat** — tempo yaml has no loudness words, and the render does not leak level |
| `eval/listen/distortion-20s/` | 0.0503 / 0.0672 / 0.0934 | distortion TF **does** move loudness |
| `eval/listen/space-20s/` | 0.1416 / 0.1263 / 0.1130 | wet is quieter |
| `eval/listen/triphop-v3-tf-raw-20s/` | 0.3496 / 0.0975 / **0.0035** | documented loudness collapse; pair also moves BPM 84 vs 128 |
| `models/dust-tf-v1/dust-tf-v1_last.json` | +1: rms **−7.5%**, centroid **+41.5%**; −0.5: rms **+6.4%**, centroid **−13.0%** | fixed-BPM pair; residual level is small vs trip-hop |

MUSIC3.md already measured the mechanism: identical recipe, trip-hop vs
dust, the **pair** moved the level channel by **68 points**; loss /
`target_mode` / `gain_penalty` moved it by single digits. *"Choose pairs
that differ in one production attribute at fixed genre/BPM/arrangement."*
Dust follows that rule. Energy-v2 and distortion TF do not.

No in-repo tempo-estimator number on energy renders (the estimator ranks
even the tempo REFs backwards; listen README: use ears).

## Smallest honest next step

Not a trainer rewrite. If the catalog should be orthogonal sliders,
**retrain the energy TF half on `prompts-cand-energy-v1.yaml`** (BPM
fixed at 110, same default `nmse`/`axis`, no attributes) and compare
against energy-slider-v2 with the existing pipeline. Distortion is the
same class of caption (BPM 140 vs 88 plus `aggressive`). Do not turn on
`--attributes` for TF.

## How to run

```bash
PYTHONPATH=. python analysis/tf_leak/run_leak.py --out docs/tf-leak
PYTHONPATH=. pytest tests/test_tf_leak.py tests/test_2d_slider_geometry.py -q
```

No Hub, no GPU, no Music 3 weights. Leak suite ~9 s CPU; both suites
~23 s. GPU trainers were not rewritten.
