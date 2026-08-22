# trip-hop in-distribution — prompt the concept, slide a little

Same lyrics and seed as `triphop-20s/`. Bases are the native 09/10 takes.
TF half-unit = 0.91, one-unit = 1.82. LM half = 0.5.

| file | sec | rms | setup |
|------|----:|----:|-------|
| `01_TRIPprompt_base.wav` | 20.02 | 0.1193 | trip-hop prompt, sliders off (native) |
| `02_TRIPprompt_TF_plus05.wav` | 20.02 | 0.1187 | trip-hop prompt + half-unit TF dust |
| `03_TRIPprompt_TF_plus1.wav` | 20.02 | 0.1174 | trip-hop prompt + one-unit TF dust |
| `04_TRIPprompt_TF_minus05.wav` | 20.02 | 0.1215 | trip-hop prompt, TF pulled toward glossy mix |
| `05_TRIPprompt_LM_plus05.wav` | 20.02 | 0.1180 | trip-hop prompt + small LM push |
| `06_TRIPprompt_both_plus05.wav` | 20.02 | 0.1163 | trip-hop prompt + half LM + half TF |
| `07_TRIPprompt_LMminus_TFplus.wav` | 20.02 | 0.1329 | trip-hop prompt, LM back a bit, TF dust |
| `08_POPprompt_base.wav` | 20.02 | 0.1385 | glossy-pop prompt, sliders off (native) |
| `09_POPprompt_TF_plus05.wav` | 20.02 | 0.1360 | pop prompt + half-unit TF dust |
| `10_POPprompt_TF_plus1.wav` | 20.02 | 0.1336 | pop prompt + one-unit TF dust |
| `11_TRIPprompt_LM05_TFx1p5.wav` | 20.02 | 0.1107 | trip-hop prompt + LM 0.5 + TF 1.5 units |
| `12_TRIPprompt_LM05_TFx2.wav` | 20.02 | 0.1064 | trip-hop prompt + LM 0.5 + TF 2 units |

Play **01 → 02 → 03 → 04** first (same trip-hop song, only the mix slider).
Then **01 → 05 → 06** (tiny LM / both on that same prompt).
Then **08 → 09 → 10** (real pop song, dusted).
Then **01 → 11 → 12** (LM 0.5 + stronger TF).

- duration: 20.0s  seed: 7
