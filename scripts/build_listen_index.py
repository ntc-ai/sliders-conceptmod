#!/usr/bin/env python3
"""Build an index.html for a listen folder tree so clips can be A/B'd in a browser.

Each subdirectory becomes a row of <audio> players, one per scale, annotated with
the run's final eval-probe numbers when a matching train jsonl is found.

    python scripts/build_listen_index.py eval/listen/loss-rank-ab-20s \
      --runs /path/to/ab /ml2/music/sliders-conceptmod/models/overnight
"""

from __future__ import annotations

import argparse
import contextlib
import html
import json
import wave
from pathlib import Path


def clip_rms(path: Path) -> float | None:
    """RMS of a 16-bit wav, so over-driven / collapsed clips are visible without
    opening them. The shipped checkpoint at raw +-2 reads 0.3496 and 0.0030
    against a 0.0975 neutral; calibrated it reads 0.1650 / 0.1124."""
    with contextlib.suppress(Exception):
        import numpy as np

        with wave.open(str(path)) as w:
            frames = w.readframes(w.getnframes())
        a = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return float(a.std())
    return None


def final_eval(run_dirs: list[Path], label: str) -> dict | None:
    """Last eval block from a train jsonl whose directory matches `label`."""
    for root in run_dirs:
        for jsonl in root.glob("*/*_train.jsonl"):
            if jsonl.parent.name != label:
                continue
            last = None
            for line in jsonl.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec.get("eval"), dict):
                    last = rec["eval"]
            return last
    return None


CSS = """
:root { color-scheme: dark light; }
body { font: 15px/1.5 ui-sans-serif, system-ui, sans-serif; margin: 0; padding: 2rem;
       background: #12131a; color: #e7e7ee; }
h1 { font-size: 1.4rem; margin: 0 0 .25rem; }
p.sub { color: #9a9aab; margin: 0 0 2rem; }
section { border: 1px solid #2a2c39; border-radius: 10px; padding: 1rem 1.25rem;
          margin-bottom: 1rem; background: #191b24; }
h2 { font-size: 1.05rem; margin: 0 0 .1rem; font-family: ui-monospace, monospace; }
.metrics { color: #9a9aab; font-size: .85rem; margin: 0 0 .75rem;
           font-family: ui-monospace, monospace; }
.metrics b { color: #7fd7a5; font-weight: 600; }
.match { font-size: .78rem; color: #12131a; background: #7fd7a5; border-radius: 5px;
         padding: .1rem .45rem; font-weight: 600; margin-left: .5rem; }
.clips { display: flex; flex-wrap: wrap; gap: .75rem; }
.clip { flex: 1 1 260px; min-width: 240px; }
.clip span { display: block; font-size: .78rem; color: #b9b9c8; margin-bottom: .25rem;
             font-family: ui-monospace, monospace; }
audio { width: 100%; }
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--runs", nargs="*", type=Path, default=[])
    ap.add_argument("--title", default=None)
    args = ap.parse_args(argv)
    root: Path = args.root
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    def ladder(wavs: list[Path]) -> dict[str, float]:
        """scale-name -> rms relative to that folder's own neutral."""
        vals = {}
        for w in wavs:
            if "slider" not in w.stem:
                continue
            key = w.stem.split("_")[-1]
            r = clip_rms(w)
            if r is not None:
                vals[key] = r
        base = vals.get("zero")
        return {k: v / base for k, v in vals.items()} if base else {}

    rows = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        wavs = sorted(sub.glob("*.wav"))
        if not wavs:
            continue
        ev = final_eval(args.runs, sub.name)
        rows.append((sub, wavs, ev, ladder(wavs)))

    # The calibrated shipped anchor at -2 is the reference level everything else
    # should be auditioned at; raw checkpoints reach it at a much lower setting.
    ref = next((l.get("minus2") for sub, _, _, l in rows
                if "calibrated" in sub.name and l.get("minus2")), None)
    matched = {}
    for sub, _, _, l in rows:
        neg = {k: v for k, v in l.items() if k.startswith("minus")}
        if ref and neg:
            best = min(neg, key=lambda k: abs(neg[k] - ref))
            matched[sub.name] = (best.replace("minus", "\u00b1"), neg[best])
    # best cos first; folders with no metrics sink to the bottom
    rows.sort(key=lambda r: (r[2] or {}).get("cos", -1), reverse=True)

    parts = [
        f"<!doctype html><meta charset=utf-8><title>{html.escape(args.title or root.name)}</title>",
        f"<style>{CSS}</style>",
        f"<h1>{html.escape(args.title or root.name)}</h1>",
        "<p class=sub>Ranked by eval-probe <code>cos</code>. Same caption, lyrics and seed "
        "throughout.<br>The <code>shipped-*</code> anchors are <b>calibrated</b>; every "
        "experimental cell is <b>raw</b> (trained <code>--no_calibrate</code>) and runs about "
        "2x hotter per step. Measured equivalence: experimental <b>&plusmn;1</b> sits where "
        "shipped <b>&plusmn;2</b> does, experimental &plusmn;0.5 where shipped &plusmn;1. "
        "Compare at equal loudness, not equal number &mdash; and note the raw cells collapse "
        "by +2 (0.10x), so that clip shows the over-drive cliff, not the slider."
        "<br><b>The comparison that matters</b> (measured equal loudness, 1.6x / 0.55x): "
        "<code>shipped-v4 @ &plusmn;2</code> vs <code>r8attn-nmse @ &plusmn;1</code> vs "
        "<code>r64attn-nmse @ &plusmn;0.5</code>. Same level, three recipes &mdash; does the "
        "higher-cos one sound more trip-hop, or just different?</p>",
    ]
    for sub, wavs, ev, _l in rows:
        parts.append("<section>")
        hint = ""
        if sub.name in matched:
            scale, ratio = matched[sub.name]
            hint = (f" <span class=match>listen at &plusmn;{html.escape(scale.lstrip(chr(177)))}"
                    f" &middot; {ratio:.2f}x</span>")
        parts.append(f"<h2>{html.escape(sub.name)}{hint}</h2>")
        if ev:
            parts.append(
                "<p class=metrics>cos <b>{:.4f}</b> &nbsp; mag {:.3f} &nbsp; "
                "proj_abs {:.4f} &nbsp; collapse {:.3f}</p>".format(
                    ev.get("cos", float("nan")), ev.get("mag", float("nan")),
                    ev.get("proj_abs", float("nan")), ev.get("collapse", float("nan")))
            )
        else:
            parts.append("<p class=metrics>(no probe metrics found)</p>")
        parts.append("<div class=clips>")
        base = clip_rms(next((w for w in wavs if "zero" in w.stem), wavs[0]))
        for wav in wavs:
            rel = f"{sub.name}/{wav.name}"
            rms = clip_rms(wav)
            note = ""
            if rms is not None:
                ratio = rms / base if base else float("nan")
                flag = " hot" if ratio > 2 else (" quiet" if ratio < 0.35 else "")
                note = f" &middot; rms {rms:.4f} ({ratio:.2f}x){flag}"
            parts.append(
                f"<div class=clip><span>{html.escape(wav.stem)}{note}</span>"
                f"<audio controls preload=none src='{html.escape(rel)}'></audio></div>"
            )
        parts.append("</div></section>")
    if not rows:
        parts.append("<section><p class=metrics>No clips rendered yet.</p></section>")

    out = root / "index.html"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {out} ({len(rows)} folders)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
