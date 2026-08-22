# Play these — MiniMax Music 3 concept sliders

Same lyrics (`I can feel it in the air tonight / Louder now or fade away`) and seed 7.

- **Slider clips** use the neutral caption. Only the LoRA scale changes.
- **REF clips** change the prompt with the slider off. That is the target sound.

Start with the **20s** folders. Those are the end-to-end long renders. `energy-30s/` is a longer-duration check of the loudness slider. The 8s folders are the earlier smoke tests (energy, distortion, gender-lm).

**Gender identity (24s, several methods):** `gender-24s/`. Transformer LoRA does not change F0; the LM slider and the condition rewrite do. Play `08`/`09` then `02`/`03`.

**Trip-hop style (20s, LM vs transformer vs both):** `triphop-20s/`. Play 01 → 02 → 04 → 06, then 07/08 to hear the axes split.

**Trip-hop in-distribution (20s):** `triphop-indist/`. Same native 09/10 takes, plus small sliders on the real trip-hop/pop prompts. Play 01 → 02 → 03 → 04, then 08 → 09.

`eval/listen/gender/` is the **failed** transformer-on-gender attempt. Use `gender-lm/` / `gender-lm-20s/` instead.

**Dust (dusty ↔ glossy), transformer only:** `models/dust-tf-v1/` is the first
TF slider that passed all six SCORING.md render gates (sidecar evidence;
weights stay out of git). No in-repo listen folder yet. Do not score an LM
half of this axis — or any LM half — with `slider_pipeline`; those gates
measure song identity the LM is *supposed* to break. Pipeline comparison
output lives under `eval/listen/pipeline/` (see
[slider_pipeline/README.md](../../slider_pipeline/README.md)).

## v3 sets (retrained with the fixed recipe)

The v3 trainers fix the off-manifold latents (x_t is anchored to generated
clean latents), train both slider directions, and antisymmetrize the LM
targets (gender-lm-v3 collapse: **−0.97**, was +0.53 — the two directions are
now nearly opposite). Multi-row prompts + `attributes` disentanglement.
Each folder also has `probe.json` from `scripts/probe_axis.py`.

| folder | axis | −2 | +2 | notes |
|---|---|---|---|---|
| `gender-v3-20s/` | singer (LM v3) | male | female | pop row; F0 on a full mix is noisy — refs barely separate |
| `gender-v3-ballad-20s/` | singer (LM v3) | male | female | sparse acoustic row, cleaner F0 reads |
| `rapslow-v3-lm-20s/` | delivery (LM v3) | slow sung ballad | rap flow | onset metric can't separate even the REFs — ears decide |
| `rapslow-v3-tf-20s/` | delivery (TF) | — | — | **negative result**: TF calibration cos_axis ≈ 0.08, unit_scale blew up to 6.7, clips overdriven. Like gender, delivery is decided in the AR LM — use the LM slider. |
| `triphop-v3-lm-20s/` | style (LM v3) | glossy pop | trip-hop | passes ref-relative centroid gate (63% of ref span) |
| `triphop-v3-tf-20s/` | style (TF, multi-row) | — | — | **negative result**: multi-row averaging kills TF axis tracking (cos_axis 0.05) |
| `triphop-v3-tf-single-20s/` + `triphop-v3-tf-raw-20s/` | style (TF v3, anchor+single) | — | — | trains better (cos 0.51) but entangles loudness: even raw ±2 slams (−) or silences (+) the mix. The app keeps **TF v2** for triphop. |
| `triphop-v3-stack-20s/` | style (TF+LM stacked) | pop | trip-hop | the app's actual `triphop` slider configuration |
| `energy-v3-lm-20s/` | energy (LM v3) | quiet | loud | **PASS**: rms rises monotonically 0.075 → 0.133, +2 lands on the loud REF |
| `energy-v3-stack-20s/` | energy (TF+LM) | quiet | loud | A/B: LM-only lands on the loud REF; the TF half at its calibrated unit (raw 4.02) overshoots |
| `energy-v3-shipped-20s/` | energy (TF+LM) | quiet | loud | **what the studio actually plays** at user ±2, via `scripts/render_shipped_slider.py` |
| `tempo-v3-lm-20s/` | tempo (LM v3) | slow | fast | trained clean (collapse −0.96); **needs ears** — no reliable automatic tempo metric |
| `distortion-v3-lm-20s/` | distortion (LM v3) | clean | heavy | trained clean (collapse −0.98); **needs ears** — crest/high-ratio cannot separate even the REFs |

Note on grading: `scripts/probe_axis.py` validates its metric against the REF
clips first. For tempo the estimator ranks the fast and slow REFs backwards, and
for distortion the two REFs are indistinguishable, so those axes report
"inconclusive — use ears" instead of a verdict. The one gate that applies
everywhere is `rms_not_destroyed` (no near-silent or slammed clips), which is
what caught the bad transformer sliders.

## 20 second sets

| folder | axis | −2 | +2 |
|---|---|---|---|
| `tempo-20s/` | tempo | slower | faster |
| `space-20s/` | room | drier / closer | wetter / more hall |
| `energy-20s/` | loudness | quieter | louder / more aggressive |
| `distortion-20s/` | guitar tone | cleaner / acoustic | more distorted / metal |
| `gender-lm-20s/` | singer (LM LoRA) | more male | more female |
| `energy-30s/` | loudness (30s proof) | quieter | louder / more aggressive |

Each folder has five wavs plus `LISTEN.md`:

1. slider −2 on the neutral prompt
2. slider 0 (base)
3. slider +2 on the neutral prompt
4. REF: plus-concept prompt, slider off
5. REF: minus-concept prompt, slider off

If a slider works, clip 3 leans toward clip 4 and clip 1 leans toward clip 5.

## 8 second smoke tests

| folder | notes |
|---|---|
| `energy/` | user-confirmed: loud works |
| `distortion/` | user-confirmed: distorted works |
| `gender-lm/` | LM LoRA; transformer `gender/` does not change gender |

## Weights

v2 family (attention-only despite `_full_` in the filename — see MUSIC3.md):

- `models/energy-slider-v2/energy_alpha8.0_rank8_full_last.safetensors`
- `models/distortion-slider/distortion_alpha8.0_rank8_full_last.safetensors`
- `models/tempo-slider/tempo_alpha8.0_rank8_full_last.safetensors`
- `models/space-slider/space_alpha8.0_rank8_full_last.safetensors`
- `models/gender-lm-slider/gender-lm_last.safetensors` (`--kind lm`)

`models/dust-tf-v1/dust-tf-v1_last.json` is the promoted full-target sidecar
(222 modules, `--loss nmse`, lr `2e-3`). The `.safetensors` is not in git.

Transformer sliders: rank-8 LoRA, 500 steps. v2 hosts `MiniMaxMusic3Attention`
only; dust-tf-v1 hosts attention + block FF + proj. Gender: rank-8 LoRA on
`Qwen3Attention`, 800 steps. All trained and rendered on GPU 0.

Regenerate or verify:

```bash
./scripts/run_gpu0_sliders.sh demo-20
./scripts/run_gpu0_sliders.sh verify
```
