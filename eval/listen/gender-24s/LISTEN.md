# 24s gender methods — why TF-only is just louder/deeper

Same lyrics, same seed. Slider clips use the **neutral** caption.
Clips are 24 seconds.

Median pitch (F0) is the tell. TF LoRA does **not** move it. The LM and
the condition rewrite do.

| play | file | F0 Hz | rms | what this is |
|-----:|------|------:|----:|---|
| 1 | `01_BASE_neutral.wav` | 156 | 0.081 | no slider |
| 2 | `02_TFlora_male_minus2.wav` | **156** | 0.073 | transformer LoRA −2 |
| 3 | `03_TFlora_female_plus2.wav` | **155** | 0.089 | transformer LoRA +2 |
| 4 | `04_CONDmean_male_minus2.wav` | 185 | 0.068 | rewrite condition mean −2 |
| 5 | `05_CONDmean_female_plus2.wav` | **218** | 0.105 | rewrite condition mean +2 |
| 6 | `08_LM_male_minus2.wav` | 189 | 0.117 | language-model LoRA −2 |
| 7 | `09_LM_female_plus2.wav` | **241** | 0.066 | language-model LoRA +2 |
| 8 | `12_REF_prompt_female.wav` | 290 | 0.098 | no slider, female prompt |

`06`/`07` are the same math as `04`/`05` (bug: mean-swap == dir-add). Skip them.
`10`/`11` stacked LM+cond and crossed; skip.

## What to hear

1. **02 vs 03** — same singer, +2 is a bit louder. F0 stuck at 156 Hz. This is the “doesn’t work” you heard.
2. **08 vs 09** — LM slider. Female is higher and brighter (F0 189 → 241, more than the TF ever moved).
3. **04 vs 05** — no LoRA. We add a gender direction to the 2048-d condition the transformer reads. F0 185 → 218. Not a full recast, but it is a pitch move the TF LoRA never made.
4. **12** — ceiling: actually prompting “woman singing” lands at 290 Hz.

## Why the transformer slider cannot swap gender

MiniMax concatenates `[latent, zeros, condition]` every step. The condition
is the encoded AR plan — that is where singer identity lives. A LoRA on
attention can remix loudness/body (you heard deeper/louder) but the
condition keeps saying “this singer” so F0 stays put.

The LM slider changes the AR tokens. The condition rewrite changes what
gets concatenated. Those are the two paths that actually move pitch.
