#!/usr/bin/env python3
"""Whisper-based blind-spot measures over slider render folders.

Two things the DSP feature set cannot see, measured with the locally cached
openai/whisper-large-v3-turbo (no network needed):

  * lyric survival: transcribe each clip; score recall of the known render
    lyrics ("I can feel it in the air tonight / Louder now or fade away").
    A slider that destroys or garbles the vocal is invisible to rms/centroid.
  * embedding distance: mean-pooled encoder hidden state (over the real,
    unpadded frames only) — cosine distance of each ladder clip to the
    same-seed scale-0 clip. A cheap "is this still the same recording?" proxy.

Controls built in: identical-file distance must be ~0; the cross-seed zero-vs-
zero distance gives the "different song, same caption" anchor.

  python scripts/blindspot_whisper.py FOLDER [FOLDER...] --out out.csv [--device cuda:0]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import wave
from pathlib import Path

import numpy as np
import torch

EXPECTED = "i can feel it in the air tonight louder now or fade away"


def load_16k(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        a = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    m = a.reshape(n, ch).mean(axis=1)
    import torchaudio.functional as AF
    return AF.resample(torch.from_numpy(m), sr, 16000).numpy()


def norm_words(s: str) -> list[str]:
    return re.sub(r"[^a-z' ]", " ", s.lower()).split()


def lyric_recall(hyp: str) -> float:
    exp = norm_words(EXPECTED)
    hw = set(norm_words(hyp))
    return sum(w in hw for w in exp) / len(exp)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folders", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--emb-out", type=Path, default=None, help="optional .npz of embeddings")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    name = "openai/whisper-large-v3-turbo"
    proc = WhisperProcessor.from_pretrained(name)
    dtype = torch.float16 if "cuda" in args.device else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(name, dtype=dtype).to(args.device).eval()

    rows, embs = [], {}
    for folder in sorted(f for f in args.folders if f.is_dir()):
        for wav in sorted(folder.glob("*.wav")):
            audio = load_16k(wav)
            feats = proc(audio, sampling_rate=16000, return_tensors="pt").input_features.to(args.device, dtype)
            with torch.no_grad():
                enc = model.model.encoder(feats).last_hidden_state[0]  # (1500, d)
                n_real = max(1, int(len(audio) / 16000 / 30.0 * enc.shape[0]))
                emb = enc[:n_real].mean(dim=0).float().cpu().numpy()
                ids = model.generate(feats, language="en", task="transcribe", max_new_tokens=64)
            text = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
            key = f"{folder.name}/{wav.name}"
            embs[key] = emb
            rows.append(dict(folder=folder.name, file=wav.name, text=text,
                             lyric_recall=round(lyric_recall(text), 3)))
            print(f"{key}: recall={rows[-1]['lyric_recall']:.2f}  {text[:70]!r}")

    # cosine distance to same-folder zero clip
    for row in rows:
        zeros = [k for k in embs if k.startswith(row["folder"] + "/") and k.endswith("_zero.wav")]
        if zeros:
            a, b = embs[f"{row['folder']}/{row['file']}"], embs[zeros[0]]
            cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
            row["emb_cos_vs_zero"] = round(cos, 4)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["folder", "file", "text", "lyric_recall", "emb_cos_vs_zero"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")
    if args.emb_out:
        np.savez(args.emb_out, **embs)
        print(f"wrote embeddings -> {args.emb_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
