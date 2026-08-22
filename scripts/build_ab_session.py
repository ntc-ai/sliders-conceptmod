#!/usr/bin/env python3
"""Build a blind listening session under eval/listen/abtest/ from ladder folders.

Reads seed folders shaped like eval/listen/pairs-4s/G-energy-seed7/ (a multiplier
ladder plus two plain-caption REF clips, all at one seed), and emits:

    abtest/
      index.html        the trial runner (serve via the existing :8901 server)
      session.json      what the page needs: trial ids, questions, hashed clip URLs
      key.json          ground truth (never fetched by the page)
      clips/<hash>.wav  hardlinked (or copied) clips with blinded names
      responses/        drop downloaded response .jsonl files here

Design (see scratchpad/listening_protocol.md):
  - direction block: 2AFC "which is more <pos-pole>?" on same-seed pairs that
    differ only in LoRA scale (extreme vs zero), 5 trials per (run, side),
    plus 6 exact repeats (self-consistency) and 4 REF-vs-REF vocabulary controls.
  - damage block: single-clip 3-point rating (fine / degraded / broken) over
    slider extremes, zero-scale + REF baselines, and 2 synthetic broken controls.

Usage:
    python scripts/build_ab_session.py            # defaults, seed 0
    python scripts/build_ab_session.py --seed 3 --out eval/listen/abtest
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import random
import re
import shutil
import time
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LISTEN = REPO / "eval" / "listen"

# one variant per caption pair; override with --runs
DEFAULT_RUNS = ["G-energy", "G-grit", "G-space", "G-vintage", "F-dust-nopen", "R-final"]
DEFAULT_ROOTS = [LISTEN / "pairs-4s", LISTEN / "curve-gp-4s"]

TRIALS_PER_SIDE = 5
N_REPEATS = 6
N_REF_CONTROLS = 4
DAMAGE_SEEDS_PER_SIDE = 2
N_DAMAGE_REFS = 4

GLOSS = {
    "Dense": ("busier, fuller arrangement", "Sparse", "emptier, more stripped back"),
    "Gritty": ("rougher, more distorted texture", "Clean", "smoother, more polished"),
    "Cavernous": ("bigger, more reverberant space", "Dry", "closer, deader room"),
    "Vintage": ("older, analog/tape character", "Modern", "contemporary hi-fi production"),
    "Dusty": ("lo-fi, worn, dusty character", "Glossy", "shiny, polished hi-fi"),
    "Trip-hop": ("slow, moody trip-hop feel", "Pop", "brighter, straighter pop feel"),
}

SCALE_RE = re.compile(r"^\d+_slider_(.+)_(minus[\d.]+|plus[\d.]+|zero)$")
REF_RE = re.compile(r"^\d+_REF_prompt_(.+)_no_slider$")


def parse_mult(tok: str) -> float:
    if tok == "zero":
        return 0.0
    sign = -1.0 if tok.startswith("minus") else 1.0
    return sign * float(tok.replace("minus", "").replace("plus", ""))


def clip_rms(path: Path) -> float | None:
    with contextlib.suppress(Exception):
        import numpy as np

        with wave.open(str(path)) as w:
            frames = w.readframes(w.getnframes())
        a = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return float(a.std())
    return None


def scan_run(roots: list[Path], run: str) -> dict:
    """-> {pos_label, neg_label, seeds: {seed: {mult: path, 'ref_pos':, 'ref_neg':}}}"""
    seeds: dict[int, dict] = {}
    pos_label = neg_label = None
    for root in roots:
        for d in sorted(root.glob(f"{run}-seed*")):
            try:
                seed = int(d.name.rsplit("seed", 1)[1])
            except ValueError:
                continue
            entry: dict = {}
            labels_by_sign: dict[str, str] = {}
            for wav in d.glob("*.wav"):
                m = SCALE_RE.match(wav.stem)
                if m:
                    mult = parse_mult(m.group(2))
                    entry[mult] = wav
                    if mult > 0:
                        labels_by_sign["pos"] = m.group(1)
                    elif mult < 0:
                        labels_by_sign["neg"] = m.group(1)
                    continue
                r = REF_RE.match(wav.stem)
                if r:
                    entry.setdefault("_refs", {})[r.group(1)] = wav
            if 0.0 not in entry:
                continue  # unusable without the zero-scale anchor
            pos_label = pos_label or labels_by_sign.get("pos")
            neg_label = neg_label or labels_by_sign.get("neg")
            seeds[seed] = entry
    if not seeds or not pos_label or not neg_label:
        raise SystemExit(f"run {run}: no usable seed folders (need zero + ladder clips)")
    for s in seeds.values():
        refs = s.pop("_refs", {})
        s["ref_pos"] = refs.get(pos_label)
        s["ref_neg"] = refs.get(neg_label)
    return {"run": run, "pos_label": pos_label, "neg_label": neg_label, "seeds": seeds}


def extreme(entry: dict, side: str) -> float | None:
    mults = [m for m in entry if isinstance(m, float)]
    pool = [m for m in mults if (m > 0 if side == "pos" else m < 0)]
    if not pool:
        return None
    return max(pool) if side == "pos" else min(pool)


def blind_name(path: Path, salt: str) -> str:
    h = hashlib.sha1(f"{salt}:{path}".encode()).hexdigest()[:12]
    return f"{h}.wav"


def question_for(pos: str) -> tuple[str, str]:
    g_pos, neg, g_neg = GLOSS.get(pos, ("", "?", ""))
    q = f"Which clip is more {pos.upper()}?"
    sub = f"{pos} = {g_pos} · {neg} = {g_neg}" if g_pos else ""
    return q, sub


def make_synthetic(zero_wav: Path, out_dir: Path, mode: str) -> Path:
    """Synthesize a known-broken control from a real clip: level collapse or silence."""
    import numpy as np

    with wave.open(str(zero_wav)) as w:
        params = w.getparams()
        frames = w.readframes(w.getnframes())
    a = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    a = a * 0.04 if mode == "collapse" else np.zeros_like(a)
    out = out_dir / f"_synth_{mode}.wav"
    with wave.open(str(out), "wb") as w:
        w.setparams(params)
        w.writeframes(a.astype(np.int16).tobytes())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="*", type=Path, default=DEFAULT_ROOTS)
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--out", type=Path, default=LISTEN / "abtest")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for trial sampling/order")
    ap.add_argument("--trials-per-side", type=int, default=TRIALS_PER_SIDE)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    session_id = f"s{args.seed}-{time.strftime('%Y%m%d')}"
    out: Path = args.out
    clips_dir = out / "clips"
    out.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(exist_ok=True)
    (out / "responses").mkdir(exist_ok=True)

    runs = [scan_run(args.roots, r) for r in args.runs]

    linked: dict[Path, str] = {}

    def link(path: Path) -> str:
        if path not in linked:
            name = blind_name(path, session_id)
            dst = clips_dir / name
            if not dst.exists():
                try:
                    import os

                    os.link(path, dst)
                except OSError:
                    shutil.copy2(path, dst)
            linked[path] = name
        return f"clips/{linked[path]}"

    trials: list[dict] = []  # public
    key: list[dict] = []  # ground truth
    n = [0]

    def tid() -> str:
        n[0] += 1
        return f"t{n[0]:03d}"

    def add_direction(run_info, seed, path_hi, mult_hi, path_lo, mult_lo, kind, repeat_of=None):
        """hi should sound more pos-pole than lo. A/B assignment randomized."""
        t = tid()
        hi_is_a = rng.random() < 0.5
        a, b = (path_hi, path_lo) if hi_is_a else (path_lo, path_hi)
        q, sub = question_for(run_info["pos_label"])
        trials.append({"id": t, "block": "direction", "kind": "trial",
                       "question": q, "sub": sub,
                       "a": link(a), "b": link(b)})
        side = "ref" if kind in ("ref_control", "practice") else ("pos" if mult_hi > 0 else "neg")
        key.append({"id": t, "block": "direction", "kind": kind,
                    "run": run_info["run"], "seed": seed,
                    "side": side,
                    "pos_label": run_info["pos_label"], "neg_label": run_info["neg_label"],
                    "mult_a": mult_hi if hi_is_a else mult_lo,
                    "mult_b": mult_lo if hi_is_a else mult_hi,
                    "correct": "a" if hi_is_a else "b",
                    "path_a": str((a).relative_to(REPO)), "path_b": str((b).relative_to(REPO)),
                    "rms_a": clip_rms(a), "rms_b": clip_rms(b),
                    "repeat_of": repeat_of})
        return t

    def add_damage(run, seed, path, mult, kind):
        t = tid()
        trials.append({"id": t, "block": "damage", "kind": "trial",
                       "question": "Rate this clip's technical quality.",
                       "sub": "", "x": link(path)})
        key.append({"id": t, "block": "damage", "kind": kind, "run": run, "seed": seed,
                    "mult": mult, "path": str(path.relative_to(REPO)) if path.is_relative_to(REPO) else path.name,
                    "rms": clip_rms(path)})
        return t

    # ---------------- direction core: (run, side) x trials_per_side ----------------
    direction_core: list[dict] = []
    for info in runs:
        for side in ("pos", "neg"):
            usable = [(s, e) for s, e in info["seeds"].items() if extreme(e, side) is not None]
            rng.shuffle(usable)
            if not usable:
                continue
            picks = [usable[i % len(usable)] for i in range(args.trials_per_side)]
            for seed, entry in picks:
                m = extreme(entry, side)
                zero = entry[0.0]
                ex = entry[m]
                # "hi" = clip whose multiplier is higher (more toward pos pole)
                if side == "pos":
                    t = add_direction(info, seed, ex, m, zero, 0.0, "core")
                else:
                    t = add_direction(info, seed, zero, 0.0, ex, m, "core")
                direction_core.append({"trial": t, "info": info, "seed": seed,
                                       "side": side, "m": m, "entry": entry})

    # ---------------- repeats: one per run, alternating sides ----------------
    by_run: dict[str, list[dict]] = {}
    for c in direction_core:
        by_run.setdefault(c["info"]["run"], []).append(c)
    repeat_pool = []
    for i, (_run, cs) in enumerate(sorted(by_run.items())):
        want = "pos" if i % 2 == 0 else "neg"
        cands = [c for c in cs if c["side"] == want] or cs
        repeat_pool.append(rng.choice(cands))
    for c in repeat_pool[:N_REPEATS]:
        info, seed, m, entry = c["info"], c["seed"], c["m"], c["entry"]
        zero, ex = entry[0.0], entry[m]
        if c["side"] == "pos":
            add_direction(info, seed, ex, m, zero, 0.0, "repeat", repeat_of=c["trial"])
        else:
            add_direction(info, seed, zero, 0.0, ex, m, "repeat", repeat_of=c["trial"])

    # ---------------- REF-vs-REF vocabulary controls ----------------
    ref_candidates = []
    for info in runs:
        for seed, entry in sorted(info["seeds"].items()):
            if entry.get("ref_pos") and entry.get("ref_neg"):
                ref_candidates.append((info, seed, entry))
    rng.shuffle(ref_candidates)
    used_runs: set[str] = set()
    picked = []
    for info, seed, entry in ref_candidates:  # prefer distinct runs
        if info["run"] in used_runs:
            continue
        picked.append((info, seed, entry))
        used_runs.add(info["run"])
        if len(picked) == N_REF_CONTROLS + 1:  # +1 for the practice trial
            break
    for i, (info, seed, entry) in enumerate(picked):
        kind = "practice" if i == 0 else "ref_control"
        t = add_direction(info, seed, entry["ref_pos"], 1.0, entry["ref_neg"], -1.0, kind)
        if kind == "practice":
            trials[-1]["kind"] = "practice"

    # ---------------- damage block ----------------
    for info in runs:
        seeds_sorted = sorted(info["seeds"].items())
        zseed, zentry = rng.choice(seeds_sorted)
        add_damage(info["run"], zseed, zentry[0.0], 0.0, "baseline_zero")
        for side in ("pos", "neg"):
            usable = [(s, e) for s, e in seeds_sorted if extreme(e, side) is not None]
            rng.shuffle(usable)
            for seed, entry in usable[:DAMAGE_SEEDS_PER_SIDE]:
                m = extreme(entry, side)
                add_damage(info["run"], seed, entry[m], m, "slider")
    ref_flat = []
    for info in runs:
        for seed, entry in sorted(info["seeds"].items()):
            for which in ("ref_pos", "ref_neg"):
                if entry.get(which):
                    ref_flat.append((info["run"], seed, entry[which]))
    rng.shuffle(ref_flat)
    for run, seed, path in ref_flat[:N_DAMAGE_REFS]:
        add_damage(run, seed, path, None, "baseline_ref")
    any_zero = next(e[0.0] for info in runs for e in info["seeds"].values() if 0.0 in e)
    for mode in ("collapse", "silence"):
        add_damage("_synth", None, make_synthetic(any_zero, clips_dir, mode), None, f"synth_{mode}")

    # ---------------- order: practice, shuffled direction, shuffled damage ----------------
    practice = [t for t in trials if t["kind"] == "practice"]
    direction = [t for t in trials if t["block"] == "direction" and t["kind"] != "practice"]
    damage = [t for t in trials if t["block"] == "damage"]
    keymap = {k["id"]: k for k in key}
    for _ in range(300):  # avoid same run twice in a row
        rng.shuffle(direction)
        if all(keymap[a["id"]]["run"] != keymap[b["id"]]["run"]
               for a, b in zip(direction, direction[1:])):
            break
    rng.shuffle(damage)
    ordered = practice + direction + damage

    # practice feedback: tell the listener which answer was 'correct'
    for t in ordered:
        if t["kind"] == "practice":
            t["feedback_correct"] = keymap[t["id"]]["correct"]

    session = {
        "session_id": session_id,
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_direction": len(direction) + len(practice),
        "n_damage": len(damage),
        "trials": ordered,
    }
    (out / "session.json").write_text(json.dumps(session, indent=1), encoding="utf-8")
    (out / "key.json").write_text(json.dumps({"session_id": session_id, "trials": key}, indent=1),
                                  encoding="utf-8")
    write_page(out)

    n_dir = len([k for k in key if k["block"] == "direction"])
    n_dam = len([k for k in key if k["block"] == "damage"])
    est = (n_dir * 13 + n_dam * 8) / 60
    print(f"wrote {out}/session.json: {n_dir} direction + {n_dam} damage trials, "
          f"{len(linked) + 2} clips, ~{est:.0f} min listening")
    print(f"open  http://localhost:8901/abtest/  (server root is eval/listen)")
    return 0


PAGE = r"""<!doctype html>
<meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>slider listening session</title>
<style>
:root { color-scheme: dark; }
body { font: 16px/1.5 ui-sans-serif, system-ui, sans-serif; margin: 0; background:#12131a;
       color:#e7e7ee; display:flex; justify-content:center; }
main { max-width: 640px; width: 100%; padding: 1.5rem; }
h1 { font-size: 1.2rem; }
.progress { height: 6px; background:#2a2c39; border-radius:3px; margin-bottom:1.2rem; }
.progress i { display:block; height:100%; background:#7fd7a5; border-radius:3px; }
.card { border:1px solid #2a2c39; background:#191b24; border-radius:12px; padding:1.5rem; }
.q { font-size:1.15rem; font-weight:600; margin:0 0 .3rem; }
.sub { color:#9a9aab; font-size:.85rem; margin:0 0 1.2rem; }
.players { display:flex; gap:1rem; margin-bottom:1.2rem; }
button { font:inherit; border:1px solid #2a2c39; background:#222430; color:#e7e7ee;
         border-radius:10px; padding:.9rem 1.2rem; cursor:pointer; flex:1; }
button:disabled { opacity:.35; cursor:default; }
button.play { font-size:1.05rem; font-weight:600; }
button.playing { border-color:#e8c56a; color:#e8c56a; }
button.played { border-color:#7fd7a5; }
.answers { display:flex; gap:1rem; }
.answers button { background:#26313a; }
.answers button:not(:disabled):hover, button.play:hover { border-color:#7fd7a5; }
.meta { color:#666a7a; font-size:.78rem; margin-top:1rem; }
.keys { color:#666a7a; font-size:.78rem; margin-top:.6rem; }
.done textarea { width:100%; height:8rem; background:#0f1016; color:#b9b9c8;
                 border:1px solid #2a2c39; border-radius:8px; font:12px ui-monospace,monospace; }
.feedback { margin-top:1rem; font-weight:600; }
a { color:#7fd7a5; }
.reset { margin-top:2rem; }
.reset a { color:#666a7a; font-size:.78rem; }
</style>
<main>
<h1 id=title>slider listening session</h1>
<div class=progress><i id=bar style="width:0"></i></div>
<div class=card id=card></div>
<div class=reset><a href="#" id=reset>restart session (wipes saved answers)</a></div>
</main>
<script>
"use strict";
let S=null, idx=0, resp=[], LS=null;
const card=document.getElementById("card"), bar=document.getElementById("bar");

fetch("session.json").then(r=>r.json()).then(s=>{ S=s; LS="abtest_"+s.session_id;
  const saved=localStorage.getItem(LS);
  if(saved){ try{ resp=JSON.parse(saved); }catch(e){ resp=[]; } }
  const done=new Set(resp.map(r=>r.trial_id));
  idx=S.trials.findIndex(t=>!done.has(t.id));
  if(idx<0) idx=S.trials.length;
  if(resp.length===0) intro(); else show();
});

function save(){ localStorage.setItem(LS, JSON.stringify(resp)); }

function intro(){
  card.innerHTML=`<p class=q>Before you start</p>
  <p>Two blocks, about 20 minutes total. Headphones on, volume set once on the
  practice clip and then left alone.</p>
  <p><b>Block 1 (pairs).</b> Each screen plays two clips, A and B — same song, same
  seed. Answer the question shown (e.g. "which is more DENSE?"). You must play both
  clips once before you can answer; replays are fine but one listen is usually
  enough. <b>If you truly cannot tell, guess</b> — forced guessing is part of the
  design, "can't tell" shows up as chance accuracy.</p>
  <p><b>Block 2 (single clips).</b> One clip per screen; rate its technical quality:
  <b>1 fine</b> (sounds like a normal render) / <b>2 degraded</b> (audibly worse:
  too quiet or loud, artifacts, smeared) / <b>3 broken</b> (silent, collapsed,
  unusable). Judge damage only, not taste.</p>
  <p>The first pair is practice with the answer revealed. Progress is saved locally;
  you can close the tab and resume.</p>
  <p class=keys>Keys — block 1: <b>q</b>/<b>w</b> play A/B, <b>1</b>/<b>2</b> answer A/B.
  Block 2: <b>space</b> play, <b>1</b>/<b>2</b>/<b>3</b> rate.</p>
  <p><button class=play id=go>Start</button></p>`;
  document.getElementById("go").onclick=show;
}

let cur=null;
function show(){
  bar.style.width=(100*Math.min(idx,S.trials.length)/S.trials.length)+"%";
  if(idx>=S.trials.length) return finish();
  const t=S.trials[idx];
  cur={t:t, t0:performance.now(), plays:{}, played:{}, audio:{}};
  if(t.block==="direction") showPair(t); else showSingle(t);
}

function mkAudio(slot, url, btn){
  const a=new Audio(url); cur.audio[slot]=a; cur.plays[slot]=0;
  a.addEventListener("ended",()=>{ cur.played[slot]=true; btn.classList.remove("playing");
    btn.classList.add("played"); update(); });
  btn.onclick=()=>{ for(const k in cur.audio){ cur.audio[k].pause(); cur.audio[k].currentTime=0;
      document.getElementById("play_"+k)?.classList.remove("playing"); }
    cur.plays[slot]++; btn.classList.add("playing"); a.currentTime=0; a.play(); };
}

function showPair(t){
  const practice = t.kind==="practice";
  card.innerHTML=`${practice?'<p class=sub><b>PRACTICE</b> — answer shown after you choose.</p>':''}
  <p class=q>${t.question}</p><p class=sub>${t.sub||""}</p>
  <div class=players>
    <button class=play id=play_a>&#9654; Play A</button>
    <button class=play id=play_b>&#9654; Play B</button>
  </div>
  <div class=answers>
    <button id=ans_a disabled>A is more</button>
    <button id=ans_b disabled>B is more</button>
  </div>
  <div class=feedback id=fb></div>
  <p class=meta>trial ${idx+1} / ${S.trials.length}</p>
  <p class=keys>q/w play · 1/2 answer</p>`;
  mkAudio("a", t.a, document.getElementById("play_a"));
  mkAudio("b", t.b, document.getElementById("play_b"));
  document.getElementById("ans_a").onclick=()=>answer("a");
  document.getElementById("ans_b").onclick=()=>answer("b");
}

function showSingle(t){
  card.innerHTML=`<p class=q>${t.question}</p>
  <p class=sub>1 = fine · 2 = degraded (audibly worse than a normal render) · 3 = broken</p>
  <div class=players><button class=play id=play_x>&#9654; Play</button></div>
  <div class=answers>
    <button id=ans_1 disabled>1 fine</button>
    <button id=ans_2 disabled>2 degraded</button>
    <button id=ans_3 disabled>3 broken</button>
  </div>
  <p class=meta>trial ${idx+1} / ${S.trials.length}</p>
  <p class=keys>space play · 1/2/3 rate</p>`;
  mkAudio("x", t.x, document.getElementById("play_x"));
  for(const v of [1,2,3]) document.getElementById("ans_"+v).onclick=()=>answer(String(v));
}

function update(){
  const t=cur.t;
  if(t.block==="direction"){
    const ok=cur.played.a&&cur.played.b;
    document.getElementById("ans_a").disabled=!ok;
    document.getElementById("ans_b").disabled=!ok;
  } else {
    for(const v of [1,2,3]) document.getElementById("ans_"+v).disabled=!cur.played.x;
  }
}

function answer(choice){
  const t=cur.t;
  if(cur.answered) return;
  if(t.kind==="practice"){
    cur.answered=true;
    const fb=document.getElementById("fb");
    fb.textContent = (choice===t.feedback_correct)
      ? "Correct — that clip was rendered with the caption on that side."
      : "The intended answer was "+t.feedback_correct.toUpperCase()+". (Practice only.)";
    fb.style.color = choice===t.feedback_correct ? "#7fd7a5" : "#e8c56a";
    setTimeout(()=>{ record(t, choice); }, 1600);
    return;
  }
  record(t, choice);
}

function record(t, choice){
  for(const k in cur.audio) cur.audio[k].pause();
  resp.push({session_id:S.session_id, trial_id:t.id, block:t.block, response:choice,
             rt_ms:Math.round(performance.now()-cur.t0), plays:cur.plays,
             ts:new Date().toISOString()});
  save(); idx++; show();
}

function finish(){
  const lines=resp.map(r=>JSON.stringify(r)).join("\n");
  const mins=resp.length?((Date.parse(resp[resp.length-1].ts)-Date.parse(resp[0].ts))/60000).toFixed(1):"?";
  card.innerHTML=`<p class=q>Done — thank you.</p>
  <p>${resp.length} responses in ~${mins} min. Save the file below into
  <code>eval/listen/abtest/responses/</code> (or anywhere) and run
  <code>scripts/score_ab_session.py</code> on it.</p>
  <p><button class=play id=dl>Download responses (.jsonl)</button></p>
  <div class=done><textarea readonly id=raw>${lines.replace(/&/g,"&amp;").replace(/</g,"&lt;")}</textarea></div>`;
  card.classList.add("done");
  document.getElementById("dl").onclick=()=>{
    const blob=new Blob([lines+"\n"],{type:"application/jsonl"});
    const a=document.createElement("a"); a.href=URL.createObjectURL(blob);
    a.download="ab_responses_"+S.session_id+"_"+Date.now()+".jsonl"; a.click(); };
}

document.getElementById("reset").onclick=(e)=>{ e.preventDefault();
  if(confirm("Wipe all saved answers for this session?")){ localStorage.removeItem(LS);
    resp=[]; idx=0; intro(); } };

document.addEventListener("keydown",(e)=>{
  if(!cur||!S||idx>=S.trials.length) return;
  const t=cur.t, k=e.key;
  if(t.block==="direction"){
    if(k==="q") document.getElementById("play_a").click();
    else if(k==="w") document.getElementById("play_b").click();
    else if(k==="1"&&!document.getElementById("ans_a").disabled) answer("a");
    else if(k==="2"&&!document.getElementById("ans_b").disabled) answer("b");
  } else {
    if(k===" "){ e.preventDefault(); document.getElementById("play_x").click(); }
    else if(["1","2","3"].includes(k)&&!document.getElementById("ans_"+k).disabled) answer(k);
  }
});
</script>
"""


def write_page(out: Path) -> None:
    (out / "index.html").write_text(PAGE, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
