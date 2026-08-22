# phase1-loss-triphop — automated comparison report

Generated 2026-08-21T16:40:24.821360. Spec: `/ml2/music/sliders-conceptmod/slider_pipeline/specs/phase1-loss-triphop.yaml`. Null E95 0.4051 over 125 seed-noise pseudo-candidates. Zero-clip consistency worst spread 0.0000.

Gates are vetoes; the score ranks gate-passers only. Paired columns are per-seed
differences against the baseline on the frozen compare seeds (mean / sd / sign-agreement);
retrain-noise floor (|mean| of floor replicates): {'rms_ln_plus': 0.0768, 'intended_proj_plus': 0.1444, 'onset_corr_plus': 0.0153, 'rms_ln_minus': 0.0287, 'intended_proj_minus': 0.1412, 'onset_corr_minus': 0.0046}.

| rank | variant | role | gates | score | Δrms_ln@+1 vs base | Δproj@+1 vs base | verdict |
|---:|---|---|---|---:|---|---|---|
| 1 | L-mse | candidate | +1 -2 +3 +4 -5 -6 +7 | 0.042 | -0.376±0.347 (100%) | +0.722±0.436 (100%) | VETOED G2_level,G5_artefacts,G6_null |
| 2 | A-alpha16 | alpha | +1 -2 +3 +4 -5 -6 +7 | 0.010 | -0.186±0.087 (100%) | +0.311±0.137 (100%) | VETOED G2_level,G5_artefacts,G6_null |
| 3 | V0-nmse | baseline | +1 -2 -3 +4 -5 -6 +7 | 0.000 | — | — | VETOED G2_level,G3_direction,G5_artefacts,G6_null |
| 4 | Vfloor-s101 | floor | +1 -2 -3 +4 -5 -6 +7 | 0.000 | +0.077±0.055 (100%) | -0.144±0.100 (100%) | VETOED G2_level,G3_direction,G5_artefacts,G6_null |
| 5 | Vfloor-s202 | floor | +1 -2 -3 +4 -5 -6 +7 | 0.000 | +0.018±0.068 (67%) | +0.016±0.068 (67%) | VETOED G2_level,G3_direction,G5_artefacts,G6_null |
| 6 | L-cos | candidate | +1 -2 -3 +4 -5 -6 +7 | 0.000 | -0.087±0.026 (100%) | +0.199±0.051 (100%) | VETOED G2_level,G3_direction,G5_artefacts,G6_null |
| 7 | L-gpen2 | control | +1 +2 -3 +4 -5 -6 +7 | 0.000 | +0.042±0.076 (67%) | -0.087±0.115 (67%) | VETOED G3_direction,G5_artefacts,G6_null |
| 8 | L-gmatch | candidate | +1 -2 -3 +4 -5 -6 +7 | 0.000 | -0.017±0.014 (100%) | -0.029±0.071 (67%) | VETOED G2_level,G3_direction,G5_artefacts,G6_null |
| 9 | L-gmatch-tw | candidate | +1 -2 -3 +4 -5 -6 +7 | 0.000 | -0.016±0.019 (100%) | -0.063±0.083 (67%) | VETOED G2_level,G3_direction,G5_artefacts,G6_null |
| 10 | L-ortho | candidate | +1 -2 -3 +4 -5 -6 +7 | 0.000 | -0.006±0.006 (100%) | +0.008±0.008 (100%) | VETOED G2_level,G3_direction,G5_artefacts,G6_null |
| 11 | A-alpha4 | alpha | +1 +2 -3 +4 +5 -6 +7 | 0.000 | +0.173±0.102 (100%) | -0.298±0.176 (100%) | VETOED G3_direction,G6_null |
| 12 | X-x0x8 | candidate | +1 -2 -3 +4 -5 -6 +7 | 0.000 | -0.003±0.046 (67%) | +0.001±0.019 (67%) | VETOED G2_level,G3_direction,G5_artefacts,G6_null |
| 13 | X-gpen2-x0x8 | candidate | +1 -2 -3 +4 -5 -6 +7 | 0.000 | +0.055±0.071 (67%) | -0.104±0.162 (67%) | VETOED G2_level,G3_direction,G5_artefacts,G6_null |

## Evidence per variant

### L-mse

- ladders: [seed 7](L-mse-seed7/), [seed 23](L-mse-seed23/), [seed 77](L-mse-seed77/)
- median intended curve: {'-1.0': -0.177, '-0.5': -0.179, '-0.25': -0.109, '0.0': 0.0, '0.25': 0.128, '0.5': 0.305, '1.0': 0.791}
- sides: {'plus': {'E': 0.3228, 'squashed': 0.139, 'scale': 1.0}, 'minus': {'E': 0.0884, 'squashed': 0.0423, 'scale': -1.0}}
- G1_silence: PASS {"offenders": [], "limit": "ratio>=0.02 and abs>=0.001"}
- G2_level: FAIL {"median_db": 8.27, "worst_db": 21.76, "limit": "median<=8.0dB worst<=14.0dB"}
- G3_direction: PASS {"spearman": 0.964, "endpoint_agree_pos": 1.0, "endpoint_agree_neg": 1.0, "limit": "rho>=0.8 agree>=0.67/side"}
- G4_identity: PASS {"per_scale": {"-1.0": {"onset_corr": 0.963, "env_corr": 0.969, "chroma_corr": 0.993}, "-0.5": {"onset_corr": 0.99, "env_corr": 0.991, "chroma_corr": 0.999}, "-0.25": {"onset_corr": 0.997, "env_corr": 0.997, "chroma_corr": 0.999}, "0.25": {"onset_corr": 0.994, "env_corr": 0.995, "chroma_corr": 1.0}, "0.5": {"onset_corr": 0.978, "env_corr": 0.971, "chroma_corr": 0.998}, "1.0": {"onset_corr": 0.91, 
- G5_artefacts: FAIL {"problems": [{"scale": 1.0, "flatness_ln": 0.899}, {"seed": 7, "scale": 1.0, "whine": true}]}
- G6_null: FAIL {"E_min": 0.0884, "null_e95": 0.4051}
- G7_overrange: PASS {"offenders": [], "limit": "no silence at over-range"}

### A-alpha16

- ladders: [seed 7](A-alpha16-seed7/), [seed 23](A-alpha16-seed23/), [seed 77](A-alpha16-seed77/)
- median intended curve: {'-1.0': -0.038, '-0.5': -0.092, '-0.25': -0.056, '0.0': 0.0, '0.25': 0.077, '0.5': 0.192, '1.0': 0.55}
- sides: {'plus': {'E': 0.2226, 'squashed': 0.1001, 'scale': 1.0}, 'minus': {'E': 0.0208, 'squashed': 0.0103, 'scale': -1.0}}
- G1_silence: PASS {"offenders": [], "limit": "ratio>=0.02 and abs>=0.001"}
- G2_level: FAIL {"median_db": 7.55, "worst_db": 17.53, "limit": "median<=8.0dB worst<=14.0dB"}
- G3_direction: PASS {"spearman": 0.893, "endpoint_agree_pos": 1.0, "endpoint_agree_neg": 0.67, "limit": "rho>=0.8 agree>=0.67/side"}
- G4_identity: PASS {"per_scale": {"-1.0": {"onset_corr": 0.97, "env_corr": 0.962, "chroma_corr": 0.995}, "-0.5": {"onset_corr": 0.991, "env_corr": 0.989, "chroma_corr": 0.999}, "-0.25": {"onset_corr": 0.996, "env_corr": 0.997, "chroma_corr": 0.999}, "0.25": {"onset_corr": 0.994, "env_corr": 0.995, "chroma_corr": 1.0}, "0.5": {"onset_corr": 0.978, "env_corr": 0.975, "chroma_corr": 0.998}, "1.0": {"onset_corr": 0.919,
- G5_artefacts: FAIL {"problems": [{"scale": 1.0, "flatness_ln": 0.687}]}
- G6_null: FAIL {"E_min": 0.0208, "null_e95": 0.4051}
- G7_overrange: PASS {"offenders": [], "limit": "no silence at over-range"}
- alpha_check: {"alpha_ratio": 2.0, "n_matched": 12, "median_abs_d_proj": 0.131, "max_abs_d_proj": 1.1224}

### V0-nmse

- ladders: [seed 7](V0-nmse-seed7/), [seed 23](V0-nmse-seed23/), [seed 77](V0-nmse-seed77/)
- median intended curve: {'-1.0': 0.108, '-0.5': -0.011, '-0.25': 0.014, '0.0': 0.0, '0.25': 0.025, '0.5': 0.054, '1.0': 0.36}
- sides: {'plus': {'E': 0.1331, 'squashed': 0.0624, 'scale': 1.0}, 'minus': {'E': 0.0, 'squashed': 0.0, 'scale': -1.0}}
- G1_silence: PASS {"offenders": [], "limit": "ratio>=0.02 and abs>=0.001"}
- G2_level: FAIL {"median_db": 6.34, "worst_db": 15.05, "limit": "median<=8.0dB worst<=14.0dB"}
- G3_direction: FAIL {"spearman": 0.429, "endpoint_agree_pos": 1.0, "endpoint_agree_neg": 0.33, "limit": "rho>=0.8 agree>=0.67/side"}
- G4_identity: PASS {"per_scale": {"-1.0": {"onset_corr": 0.97, "env_corr": 0.966, "chroma_corr": 0.996}, "-0.5": {"onset_corr": 0.989, "env_corr": 0.991, "chroma_corr": 0.999}, "-0.25": {"onset_corr": 0.997, "env_corr": 0.998, "chroma_corr": 1.0}, "0.25": {"onset_corr": 0.995, "env_corr": 0.996, "chroma_corr": 1.0}, "0.5": {"onset_corr": 0.982, "env_corr": 0.983, "chroma_corr": 0.999}, "1.0": {"onset_corr": 0.936, "
- G5_artefacts: FAIL {"problems": [{"scale": 1.0, "flatness_ln": 0.548}]}
- G6_null: FAIL {"E_min": 0.0, "null_e95": 0.4051}
- G7_overrange: PASS {"offenders": [], "limit": "no silence at over-range"}

### Vfloor-s101

- ladders: [seed 7](Vfloor-s101-seed7/), [seed 23](Vfloor-s101-seed23/), [seed 77](Vfloor-s101-seed77/)
- median intended curve: {'-1.0': 0.11, '-0.5': -0.002, '-0.25': 0.017, '0.0': 0.0, '0.25': -0.013, '0.5': 0.003, '1.0': 0.305}
- sides: {'plus': {'E': 0.1216, 'squashed': 0.0573, 'scale': 1.0}, 'minus': {'E': 0.0, 'squashed': 0.0, 'scale': -1.0}}
- G1_silence: PASS {"offenders": [], "limit": "ratio>=0.02 and abs>=0.001"}
- G2_level: FAIL {"median_db": 5.51, "worst_db": 14.0, "limit": "median<=8.0dB worst<=14.0dB"}
- G3_direction: FAIL {"spearman": 0.107, "endpoint_agree_pos": 1.0, "endpoint_agree_neg": 0.33, "limit": "rho>=0.8 agree>=0.67/side"}
- G4_identity: PASS {"per_scale": {"-1.0": {"onset_corr": 0.966, "env_corr": 0.968, "chroma_corr": 0.995}, "-0.5": {"onset_corr": 0.99, "env_corr": 0.992, "chroma_corr": 0.999}, "-0.25": {"onset_corr": 0.997, "env_corr": 0.998, "chroma_corr": 0.999}, "0.25": {"onset_corr": 0.996, "env_corr": 0.997, "chroma_corr": 0.999}, "0.5": {"onset_corr": 0.984, "env_corr": 0.985, "chroma_corr": 0.999}, "1.0": {"onset_corr": 0.94
- G5_artefacts: FAIL {"problems": [{"scale": 1.0, "flatness_ln": 0.414}]}
- G6_null: FAIL {"E_min": 0.0, "null_e95": 0.4051}
- G7_overrange: PASS {"offenders": [], "limit": "no silence at over-range"}

### Vfloor-s202

- ladders: [seed 7](Vfloor-s202-seed7/), [seed 23](Vfloor-s202-seed23/), [seed 77](Vfloor-s202-seed77/)
- median intended curve: {'-1.0': 0.098, '-0.5': -0.023, '-0.25': -0.005, '0.0': 0.0, '0.25': 0.033, '0.5': 0.093, '1.0': 0.413}
- sides: {'plus': {'E': 0.1392, 'squashed': 0.0651, 'scale': 1.0}, 'minus': {'E': 0.0, 'squashed': 0.0, 'scale': -1.0}}
- G1_silence: PASS {"offenders": [], "limit": "ratio>=0.02 and abs>=0.001"}
- G2_level: FAIL {"median_db": 6.22, "worst_db": 14.28, "limit": "median<=8.0dB worst<=14.0dB"}
- G3_direction: FAIL {"spearman": 0.464, "endpoint_agree_pos": 1.0, "endpoint_agree_neg": 0.33, "limit": "rho>=0.8 agree>=0.67/side"}
- G4_identity: PASS {"per_scale": {"-1.0": {"onset_corr": 0.963, "env_corr": 0.967, "chroma_corr": 0.995}, "-0.5": {"onset_corr": 0.99, "env_corr": 0.991, "chroma_corr": 0.999}, "-0.25": {"onset_corr": 0.996, "env_corr": 0.997, "chroma_corr": 0.999}, "0.25": {"onset_corr": 0.995, "env_corr": 0.996, "chroma_corr": 1.0}, "0.5": {"onset_corr": 0.981, "env_corr": 0.982, "chroma_corr": 0.999}, "1.0": {"onset_corr": 0.933,
- G5_artefacts: FAIL {"problems": [{"scale": 1.0, "flatness_ln": 0.57}]}
- G6_null: FAIL {"E_min": 0.0, "null_e95": 0.4051}
- G7_overrange: PASS {"offenders": [], "limit": "no silence at over-range"}

### L-cos

- ladders: [seed 7](L-cos-seed7/), [seed 23](L-cos-seed23/), [seed 77](L-cos-seed77/)
- median intended curve: {'-1.0': 0.024, '-0.5': -0.055, '-0.25': -0.043, '0.0': 0.0, '0.25': 0.073, '0.5': 0.183, '1.0': 0.54}
- sides: {'plus': {'E': 0.2392, 'squashed': 0.1068, 'scale': 1.0}, 'minus': {'E': 0.0, 'squashed': 0.0, 'scale': -1.0}}
- G1_silence: PASS {"offenders": [], "limit": "ratio>=0.02 and abs>=0.001"}
- G2_level: FAIL {"median_db": 6.99, "worst_db": 16.07, "limit": "median<=8.0dB worst<=14.0dB"}
- G3_direction: FAIL {"spearman": 0.786, "endpoint_agree_pos": 1.0, "endpoint_agree_neg": 0.33, "limit": "rho>=0.8 agree>=0.67/side"}
- G4_identity: PASS {"per_scale": {"-1.0": {"onset_corr": 0.967, "env_corr": 0.966, "chroma_corr": 0.994}, "-0.5": {"onset_corr": 0.991, "env_corr": 0.991, "chroma_corr": 0.998}, "-0.25": {"onset_corr": 0.997, "env_corr": 0.998, "chroma_corr": 0.999}, "0.25": {"onset_corr": 0.995, "env_corr": 0.996, "chroma_corr": 0.999}, "0.5": {"onset_corr": 0.979, "env_corr": 0.982, "chroma_corr": 0.999}, "1.0": {"onset_corr": 0.9
- G5_artefacts: FAIL {"problems": [{"scale": 1.0, "flatness_ln": 0.794}]}
- G6_null: FAIL {"E_min": 0.0, "null_e95": 0.4051}
- G7_overrange: PASS {"offenders": [], "limit": "no silence at over-range"}

### L-gpen2

- ladders: [seed 7](L-gpen2-seed7/), [seed 23](L-gpen2-seed23/), [seed 77](L-gpen2-seed77/)
- median intended curve: {'-1.0': 0.102, '-0.5': -0.008, '-0.25': 0.001, '0.0': 0.0, '0.25': 0.028, '0.5': 0.077, '1.0': 0.3}
- sides: {'plus': {'E': 0.1375, 'squashed': 0.0643, 'scale': 1.0}, 'minus': {'E': 0.0, 'squashed': 0.0, 'scale': -1.0}}
- G1_silence: PASS {"offenders": [], "limit": "ratio>=0.02 and abs>=0.001"}
- G2_level: PASS {"median_db": 6.27, "worst_db": 13.93, "limit": "median<=8.0dB worst<=14.0dB"}
- G3_direction: FAIL {"spearman": 0.429, "endpoint_agree_pos": 1.0, "endpoint_agree_neg": 0.33, "limit": "rho>=0.8 agree>=0.67/side"}
- G4_identity: PASS {"per_scale": {"-1.0": {"onset_corr": 0.978, "env_corr": 0.969, "chroma_corr": 0.997}, "-0.5": {"onset_corr": 0.992, "env_corr": 0.991, "chroma_corr": 0.999}, "-0.25": {"onset_corr": 0.997, "env_corr": 0.998, "chroma_corr": 0.999}, "0.25": {"onset_corr": 0.996, "env_corr": 0.997, "chroma_corr": 1.0}, "0.5": {"onset_corr": 0.984, "env_corr": 0.985, "chroma_corr": 0.999}, "1.0": {"onset_corr": 0.946
- G5_artefacts: FAIL {"problems": [{"scale": 1.0, "flatness_ln": 0.413}]}
- G6_null: FAIL {"E_min": 0.0, "null_e95": 0.4051}
- G7_overrange: PASS {"offenders": [], "limit": "no silence at over-range"}

### L-gmatch

- ladders: [seed 7](L-gmatch-seed7/), [seed 23](L-gmatch-seed23/), [seed 77](L-gmatch-seed77/)
- median intended curve: {'-1.0': 0.143, '-0.5': 0.002, '-0.25': 0.015, '0.0': 0.0, '0.25': 0.01, '0.5': 0.034, '1.0': 0.31}
- sides: {'plus': {'E': 0.1112, 'squashed': 0.0527, 'scale': 1.0}, 'minus': {'E': 0.0, 'squashed': 0.0, 'scale': -1.0}}
- G1_silence: PASS {"offenders": [], "limit": "ratio>=0.02 and abs>=0.001"}
- G2_level: FAIL {"median_db": 6.44, "worst_db": 15.33, "limit": "median<=8.0dB worst<=14.0dB"}
- G3_direction: FAIL {"spearman": 0.286, "endpoint_agree_pos": 1.0, "endpoint_agree_neg": 0.33, "limit": "rho>=0.8 agree>=0.67/side"}
- G4_identity: PASS {"per_scale": {"-1.0": {"onset_corr": 0.966, "env_corr": 0.968, "chroma_corr": 0.996}, "-0.5": {"onset_corr": 0.99, "env_corr": 0.991, "chroma_corr": 0.999}, "-0.25": {"onset_corr": 0.997, "env_corr": 0.998, "chroma_corr": 1.0}, "0.25": {"onset_corr": 0.995, "env_corr": 0.996, "chroma_corr": 1.0}, "0.5": {"onset_corr": 0.981, "env_corr": 0.982, "chroma_corr": 0.999}, "1.0": {"onset_corr": 0.929, "
- G5_artefacts: FAIL {"problems": [{"scale": 1.0, "flatness_ln": 0.414}]}
- G6_null: FAIL {"E_min": 0.0, "null_e95": 0.4051}
- G7_overrange: PASS {"offenders": [], "limit": "no silence at over-range"}

### L-gmatch-tw

- ladders: [seed 7](L-gmatch-tw-seed7/), [seed 23](L-gmatch-tw-seed23/), [seed 77](L-gmatch-tw-seed77/)
- median intended curve: {'-1.0': 0.17, '-0.5': 0.018, '-0.25': 0.013, '0.0': 0.0, '0.25': -0.004, '0.5': 0.014, '1.0': 0.283}
- sides: {'plus': {'E': 0.1006, 'squashed': 0.0479, 'scale': 1.0}, 'minus': {'E': 0.0, 'squashed': 0.0, 'scale': -1.0}}
- G1_silence: PASS {"offenders": [], "limit": "ratio>=0.02 and abs>=0.001"}
- G2_level: FAIL {"median_db": 6.35, "worst_db": 15.37, "limit": "median<=8.0dB worst<=14.0dB"}
- G3_direction: FAIL {"spearman": -0.036, "endpoint_agree_pos": 1.0, "endpoint_agree_neg": 0.33, "limit": "rho>=0.8 agree>=0.67/side"}
- G4_identity: PASS {"per_scale": {"-1.0": {"onset_corr": 0.962, "env_corr": 0.969, "chroma_corr": 0.996}, "-0.5": {"onset_corr": 0.988, "env_corr": 0.992, "chroma_corr": 0.998}, "-0.25": {"onset_corr": 0.996, "env_corr": 0.998, "chroma_corr": 1.0}, "0.25": {"onset_corr": 0.995, "env_corr": 0.996, "chroma_corr": 1.0}, "0.5": {"onset_corr": 0.979, "env_corr": 0.983, "chroma_corr": 0.999}, "1.0": {"onset_corr": 0.932, 
- G5_artefacts: FAIL {"problems": [{"scale": 1.0, "flatness_ln": 0.381}]}
- G6_null: FAIL {"E_min": 0.0, "null_e95": 0.4051}
- G7_overrange: PASS {"offenders": [], "limit": "no silence at over-range"}

### L-ortho

- ladders: [seed 7](L-ortho-seed7/), [seed 23](L-ortho-seed23/), [seed 77](L-ortho-seed77/)
- median intended curve: {'-1.0': 0.097, '-0.5': -0.013, '-0.25': 0.001, '0.0': 0.0, '0.25': 0.025, '0.5': 0.067, '1.0': 0.378}
- sides: {'plus': {'E': 0.1359, 'squashed': 0.0636, 'scale': 1.0}, 'minus': {'E': 0.0, 'squashed': 0.0, 'scale': -1.0}}
- G1_silence: PASS {"offenders": [], "limit": "ratio>=0.02 and abs>=0.001"}
- G2_level: FAIL {"median_db": 6.36, "worst_db": 15.17, "limit": "median<=8.0dB worst<=14.0dB"}
- G3_direction: FAIL {"spearman": 0.429, "endpoint_agree_pos": 1.0, "endpoint_agree_neg": 0.33, "limit": "rho>=0.8 agree>=0.67/side"}
- G4_identity: PASS {"per_scale": {"-1.0": {"onset_corr": 0.97, "env_corr": 0.966, "chroma_corr": 0.996}, "-0.5": {"onset_corr": 0.99, "env_corr": 0.991, "chroma_corr": 0.999}, "-0.25": {"onset_corr": 0.997, "env_corr": 0.998, "chroma_corr": 0.999}, "0.25": {"onset_corr": 0.995, "env_corr": 0.996, "chroma_corr": 1.0}, "0.5": {"onset_corr": 0.981, "env_corr": 0.983, "chroma_corr": 0.999}, "1.0": {"onset_corr": 0.937, 
- G5_artefacts: FAIL {"problems": [{"scale": 1.0, "flatness_ln": 0.535}]}
- G6_null: FAIL {"E_min": 0.0, "null_e95": 0.4051}
- G7_overrange: PASS {"offenders": [], "limit": "no silence at over-range"}

### A-alpha4

- ladders: [seed 7](A-alpha4-seed7/), [seed 23](A-alpha4-seed23/), [seed 77](A-alpha4-seed77/)
- median intended curve: {'-1.0': 0.149, '-0.5': 0.018, '-0.25': 0.027, '0.0': 0.0, '0.25': -0.022, '0.5': -0.044, '1.0': 0.193}
- sides: {'plus': {'E': 0.0796, 'squashed': 0.0383, 'scale': 1.0}, 'minus': {'E': 0.0, 'squashed': 0.0, 'scale': -1.0}}
- G1_silence: PASS {"offenders": [], "limit": "ratio>=0.02 and abs>=0.001"}
- G2_level: PASS {"median_db": 5.24, "worst_db": 12.53, "limit": "median<=8.0dB worst<=14.0dB"}
- G3_direction: FAIL {"spearman": -0.214, "endpoint_agree_pos": 1.0, "endpoint_agree_neg": 0.33, "limit": "rho>=0.8 agree>=0.67/side"}
- G4_identity: PASS {"per_scale": {"-1.0": {"onset_corr": 0.973, "env_corr": 0.977, "chroma_corr": 0.996}, "-0.5": {"onset_corr": 0.991, "env_corr": 0.994, "chroma_corr": 0.999}, "-0.25": {"onset_corr": 0.997, "env_corr": 0.998, "chroma_corr": 1.0}, "0.25": {"onset_corr": 0.996, "env_corr": 0.998, "chroma_corr": 0.999}, "0.5": {"onset_corr": 0.985, "env_corr": 0.989, "chroma_corr": 0.999}, "1.0": {"onset_corr": 0.948
- G5_artefacts: PASS {"problems": []}
- G6_null: FAIL {"E_min": 0.0, "null_e95": 0.4051}
- G7_overrange: PASS {"offenders": [], "limit": "no silence at over-range"}
- alpha_check: {"alpha_ratio": 0.5, "n_matched": 12, "median_abs_d_proj": 0.1345, "max_abs_d_proj": 0.8348}

### X-x0x8

- ladders: [seed 7](X-x0x8-seed7/), [seed 23](X-x0x8-seed23/), [seed 77](X-x0x8-seed77/)
- median intended curve: {'-1.0': 0.103, '-0.5': -0.015, '-0.25': -0.0, '0.0': 0.0, '0.25': 0.035, '0.5': 0.078, '1.0': 0.34}
- sides: {'plus': {'E': 0.1285, 'squashed': 0.0604, 'scale': 1.0}, 'minus': {'E': 0.0, 'squashed': 0.0, 'scale': -1.0}}
- G1_silence: PASS {"offenders": [], "limit": "ratio>=0.02 and abs>=0.001"}
- G2_level: FAIL {"median_db": 5.92, "worst_db": 15.41, "limit": "median<=8.0dB worst<=14.0dB"}
- G3_direction: FAIL {"spearman": 0.464, "endpoint_agree_pos": 1.0, "endpoint_agree_neg": 0.33, "limit": "rho>=0.8 agree>=0.67/side"}
- G4_identity: PASS {"per_scale": {"-1.0": {"onset_corr": 0.968, "env_corr": 0.966, "chroma_corr": 0.994}, "-0.5": {"onset_corr": 0.99, "env_corr": 0.991, "chroma_corr": 0.999}, "-0.25": {"onset_corr": 0.996, "env_corr": 0.998, "chroma_corr": 0.999}, "0.25": {"onset_corr": 0.995, "env_corr": 0.997, "chroma_corr": 0.999}, "0.5": {"onset_corr": 0.982, "env_corr": 0.983, "chroma_corr": 0.999}, "1.0": {"onset_corr": 0.93
- G5_artefacts: FAIL {"problems": [{"scale": 1.0, "flatness_ln": 0.447}]}
- G6_null: FAIL {"E_min": 0.0, "null_e95": 0.4051}
- G7_overrange: PASS {"offenders": [], "limit": "no silence at over-range"}

### X-gpen2-x0x8

- ladders: [seed 7](X-gpen2-x0x8-seed7/), [seed 23](X-gpen2-x0x8-seed23/), [seed 77](X-gpen2-x0x8-seed77/)
- median intended curve: {'-1.0': 0.101, '-0.5': -0.013, '-0.25': -0.008, '0.0': 0.0, '0.25': 0.038, '0.5': 0.1, '1.0': 0.298}
- sides: {'plus': {'E': 0.1465, 'squashed': 0.0682, 'scale': 1.0}, 'minus': {'E': 0.0, 'squashed': 0.0, 'scale': -1.0}}
- G1_silence: PASS {"offenders": [], "limit": "ratio>=0.02 and abs>=0.001"}
- G2_level: FAIL {"median_db": 5.73, "worst_db": 14.03, "limit": "median<=8.0dB worst<=14.0dB"}
- G3_direction: FAIL {"spearman": 0.464, "endpoint_agree_pos": 1.0, "endpoint_agree_neg": 0.33, "limit": "rho>=0.8 agree>=0.67/side"}
- G4_identity: PASS {"per_scale": {"-1.0": {"onset_corr": 0.974, "env_corr": 0.97, "chroma_corr": 0.996}, "-0.5": {"onset_corr": 0.993, "env_corr": 0.991, "chroma_corr": 0.999}, "-0.25": {"onset_corr": 0.997, "env_corr": 0.998, "chroma_corr": 0.999}, "0.25": {"onset_corr": 0.996, "env_corr": 0.997, "chroma_corr": 0.999}, "0.5": {"onset_corr": 0.986, "env_corr": 0.985, "chroma_corr": 0.998}, "1.0": {"onset_corr": 0.94
- G5_artefacts: FAIL {"problems": [{"scale": 1.0, "flatness_ln": 0.329}]}
- G6_null: FAIL {"E_min": 0.0, "null_e95": 0.4051}
- G7_overrange: PASS {"offenders": [], "limit": "no silence at over-range"}
