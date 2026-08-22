#!/usr/bin/env python3
"""One song. Every shipped slider ramps in sequence.

LM halves: gain follows the envelope over AR *frames* (one step = one audio
frame; KV-cache forwards see a single token, so we set a scalar per step).
Transformer halves: each denoise chunk gets the matching slice of the same
envelope, interpolated onto that chunk's latent frames.

Inactive sliders sit at 0. One generate, one seed, one plan that actually
moves.

  CUDA_VISIBLE_DEVICES=1 $PY scripts/render_song_ramp.py \
      --out eval/listen/song-ramp-60s/song.wav --duration 60 --device 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = Path("/ml2/music")


def _early_visible_device(argv: list[str]) -> str:
    for i, arg in enumerate(argv):
        if arg == "--device" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--device="):
            return arg.split("=", 1)[1]
    return os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]


os.environ["CUDA_VISIBLE_DEVICES"] = _early_visible_device(sys.argv[1:])
os.environ.setdefault("HF_HOME", "/ml2/music/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

for path in (str(ROOT), str(APP_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch  # noqa: E402

from conceptmod.textsliders.generate_gender_stack import _apply, _wrap_sidecar  # noqa: E402
from conceptmod.textsliders.generate_listen import DEFAULT_MODEL_DIR, _write_wav  # noqa: E402
from conceptmod.textsliders.infer_music3 import _load_pipeline  # noqa: E402

# Shown in this order. Each slot: hold at -1, sweep to +1, hold at +1.
SLIDER_ORDER = [
    "energy",
    "distortion",
    "tempo",
    "space",
    "gender",
    "triphop",
    "rapslow",
    "live",
    "breath",
    "rhyme",
]

SLOT = 9.5
FADE_IN = 0.5   # 0 -> -1: slots enter through 0, no boundary jump
HOLD_LO = 3.0   # sit on the -1 pole; the first ~2s also drain the previous
                # slot's audio out of the rolling context
SWEEP = 1.0
HOLD_HI = 4.0   # the +1 pole needs a long dwell: identity axes only settle
                # once the rolling context is pure-pole audio
FADE_OUT = 1.0  # +1 -> 0: release before the next slider takes over

# The caption is assembled from named parts so a pole variant can replace
# exactly the sentences it owns and keep every other anchor (BPM, structure,
# vocal continuity) — that's what keeps the words and the beat coherent while
# the caption follows the fader.
CAP = {
    "genre": "Genre: pop. BPM 110.",
    "energy": "Medium energy, even mood.",
    "mix": "Dry-ish balanced studio mix.",
    "who": "One clear lead vocal mixed upfront, ordinary timbre, even phrasing, no backing vocals.",
    "style": "Sung, not rapped.",
    "cont": "The singer starts in the first bar and stays almost continuous to the end.",
    "arr": "Verse-chorus pop. Drums, bass and guitar sit behind the vocal at a comfortable level. "
           "Vocals carry the song; instrumental gaps are short.",
}


def build_caption(overrides: dict | None = None) -> str:
    c = dict(CAP)
    c.update(overrides or {})
    return (
        "Global Metadata:\n"
        f"{c['genre']} {c['energy']} {c['mix']}\n"
        "Vocal Details:\n"
        f"{c['who']} {c['style']} {c['cont']}\n"
        "Arrangement:\n"
        f"{c['arr']}"
    )


NEUTRAL = build_caption()
LYRICS = (
    "[verse]\n"
    "I found a window facing west\n"
    "A half-made melody at rest\n"
    "I hummed the part you didn't write\n"
    "And left it better than last night\n"
    "[chorus]\n"
    "Leave it open, leave it on\n"
    "Held by every other hand\n"
    "A better world than we had planned\n"
    "Leave it open, leave it on\n"
    "[verse]\n"
    "Down every alley, same old tune\n"
    "A stranger whistling it at noon\n"
    "I fold my verse into the wind\n"
    "You send a better one back in\n"
    "[chorus]\n"
    "Leave it open, leave it on\n"
    "Held by every other hand\n"
    "A better world than we had planned\n"
    "Leave it open, leave it on\n"
    "[bridge]\n"
    "Nobody owns the morning light\n"
    "We pass it hand to hand at night\n"
    "The weights you set down I can lift\n"
    "And that's the only kind of gift\n"
    "[chorus]\n"
    "Leave it open, leave it on\n"
    "Held by every other hand\n"
    "A better world than we had planned\n"
    "Leave it open, leave it on\n"
    "[verse]\n"
    "I left the latch off of the gate\n"
    "For anyone still up this late\n"
    "I set my weights down on the stair\n"
    "You built a better one from there\n"
    "[chorus]\n"
    "Leave it open, leave it on\n"
    "Held by every other hand\n"
    "A better world than we had planned\n"
    "Leave it open, leave it on"
)


# Per-pole caption overrides (minus, plus), keywords distilled from each
# slider's training prompts. With --dynamic-prompt the KV rebuild swaps in
# build_caption(override) while the fader sits on that pole: only the
# sentences the pole owns change (stated positively, and NEUTRAL wording
# that would contradict the pole is what gets replaced); BPM, song
# structure and vocal continuity stay, so words and beat stay coherent.
POLE_OVERRIDES = {
    "energy": (
        {"energy": "Very low energy, hushed and calm, a whispered intimate vocal over a quiet sparse backing."},
        {"energy": "Very high energy, loud and aggressive, pounding drums, shouted vocal, dense slammed mix."},
    ),
    "distortion": (
        {"arr": "Verse-chorus pop. Clean unplugged acoustic guitars, bass and drums sit behind the vocal. "
                "Vocals carry the song; instrumental gaps are short."},
        {"arr": "Verse-chorus pop. Heavily distorted electric guitars, thick fuzz and crushing overdrive "
                "behind the vocal. Vocals carry the song; instrumental gaps are short."},
    ),
    "tempo": (
        {"genre": "Genre: pop. BPM 60.", "energy": "Very slow dragged tempo, spacious half-time feel."},
        {"genre": "Genre: pop. BPM 175.", "energy": "Very fast racing tempo, frantic double-time drums."},
    ),
    "space": (
        {"mix": "Bone dry close-mic mix, dead room, intimate booth."},
        {"mix": "Huge cathedral reverb, vast hall, long wet tails, the band washed in echo."},
    ),
    "gender": (
        {"who": "One male lead singer mixed upfront. A man is singing, his voice is deep and masculine. "
                "Even phrasing, no backing vocals."},
        {"who": "One female lead singer mixed upfront. A woman is singing, her voice is feminine. "
                "Even phrasing, no backing vocals."},
    ),
    "triphop": (
        {"genre": "Genre: glossy radio pop. BPM 110.",
         "mix": "Sparkling polished mix, punchy bright drums, shiny compressed vocal."},
        {"genre": "Genre: trip-hop. BPM 84.",
         "mix": "Dusty slow breakbeats, vinyl crackle, smoky nocturnal mix, dark minor keys."},
    ),
    "rapslow": (
        {"style": "The vocalist sings long sustained melodic notes with slow emotional ballad delivery."},
        {"style": "The vocalist raps fast rhythmic spoken bars with a confident syncopated flow."},
    ),
    "live": (
        {"mix": "Sealed precise studio mix, tracked alone in a booth, click-tight."},
        {"mix": "Live in a small club, room mics, drum-kit bleed on the vocal mic, an audience in the room."},
    ),
    "breath": (
        {"who": "One clear lead vocal mixed upfront, belted clean timbre, compressed and polished, "
                "even phrasing, no backing vocals."},
        {"who": "One close-mic breathy lead vocal, every inhale audible, hushed intimate timbre, "
                "even phrasing, no backing vocals."},
    ),
    "rhyme": (
        {"style": "Sung, not rapped, in free-flowing through-composed lines."},
        {"style": "Sung, not rapped, in tight AABB couplets where every line-ending rhymes and lands hard."},
    ),
}


def _smooth(u: float) -> float:
    u = max(0.0, min(1.0, u))
    return u * u * (3 - 2 * u)


def slot_envelope(n: int, slider_index: int, n_sliders: int) -> torch.Tensor:
    """User-scale over AR frames. 0 outside this slider's slot; enters and
    exits through 0 so slot boundaries do not jump."""
    env = torch.zeros(n)
    slot = n / n_sliders
    t0 = slider_index * slot
    a = t0 + slot * (FADE_IN / SLOT)
    b = a + slot * (HOLD_LO / SLOT)
    c = b + slot * (SWEEP / SLOT)
    d = c + slot * (HOLD_HI / SLOT)
    t1 = t0 + slot
    for i in range(n):
        t = float(i)
        if t < t0 or t >= t1:
            continue
        if t < a:
            env[i] = -_smooth((t - t0) / max(a - t0, 1.0))
        elif t < b:
            env[i] = -1.0
        elif t < c:
            env[i] = -1.0 + 2.0 * _smooth((t - b) / max(c - b, 1.0))
        elif t < d:
            env[i] = 1.0
        else:
            env[i] = 1.0 - _smooth((t - d) / max(t1 - d, 1.0))
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--sliders", default=",".join(SLIDER_ORDER))
    parser.add_argument(
        "--refresh", type=int, default=15,
        help="AR frames between full-prefix KV rebuilds (cached AR never re-reads old keys otherwise)",
    )
    parser.add_argument(
        "--flat", type=float, default=None,
        help="debug: hold every requested slider at this constant user-scale gain instead of the envelope",
    )
    parser.add_argument(
        "--dynamic-prompt", action="store_true",
        help="at each KV rebuild, swap the caption for a variant carrying the active pole's "
        "training keywords so the text conditioning pulls with the LoRA gain",
    )
    parser.add_argument(
        "--prompt-gate", type=float, default=0.35,
        help="|gain| above which the pole's keyword caption replaces the neutral one",
    )
    parser.add_argument(
        "--context-window", type=float, default=2.0,
        help="seconds of audio history kept at each KV rebuild; identity axes (gender, live, breath) "
        "cannot flip mid-song against a long anchored history. 0 = keep everything",
    )
    args = parser.parse_args()

    from app import sliders as registry
    from diffusers.modular_pipelines.minimax_music3.denoise import MiniMaxMusic3ChunkConditionStep
    from diffusers.modular_pipelines.minimax_music3.encoders import (
        MiniMaxMusic3SemanticGenerationStep,
        _AR_CFG_SCALE,
        _AR_CFG_TOP_K,
        _AUDIO_CODE_OFFSET,
        _AUDIO_END_TOKEN_ID,
        _MAX_AUDIO_FRAMES,
        _SEMANTIC_VOCAB_SIZE,
        _embed_audio_frame,
        _generate_depth_codes,
        _sample_top_k,
    )

    slider_ids = [s.strip() for s in args.sliders.split(",") if s.strip()]
    catalog = {s["id"]: s for s in registry.catalog()["sliders"]}
    missing = [s for s in slider_ids if s not in catalog]
    if missing:
        raise SystemExit(f"unknown sliders: {missing}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda:0"
    pipe = _load_pipeline(Path(args.model_dir), device)
    frame_rate = float(pipe.frame_rate)
    max_frames = int(args.duration * frame_rate)
    print(f"frame_rate={frame_rate} max_frames={max_frames}", flush=True)

    # One envelope per slider, then attach to each resolved component.
    lm_nets: list[tuple[str, object, torch.Tensor, float]] = []
    tf_nets: list[tuple[str, object, torch.Tensor, float]] = []
    slider_envs: list[tuple[str, torch.Tensor]] = []
    apply_pairs = []
    schedule = []

    for i, sid in enumerate(slider_ids):
        env = slot_envelope(max_frames, i, len(slider_ids))
        if args.flat is not None:
            env = torch.full((max_frames,), float(args.flat))
        slider_envs.append((sid, env))
        comps = registry.resolve([{"id": sid, "scale": 1.0}])
        entry = catalog[sid]
        schedule.append(
            {
                "id": sid,
                "name": sid.upper(),
                "minus": entry["label_minus"].upper(),
                "plus": entry["label_plus"].upper(),
                "t0": i * SLOT,
                "t1": (i + 1) * SLOT,
                "fade_in": FADE_IN,
                "hold_lo": HOLD_LO,
                "sweep": SWEEP,
                "hold_hi": HOLD_HI,
                "fade_out": FADE_OUT,
            }
        )
        for comp in comps:
            net, _meta = _wrap_sidecar(pipe, device, Path(comp["weights"]), comp["kind"])
            unit = float(comp["multiplier"])
            net.set_lora_slider(unit)
            apply_pairs.append((net, unit))
            rec = (sid, net, env, unit)
            if comp["kind"] == "language_model":
                lm_nets.append(rec)
            else:
                tf_nets.append(rec)
            print(
                f"{sid:12} {comp['kind']:16} {Path(comp['weights']).name} unit={unit:+.3f}",
                flush=True,
            )

    def set_lm_scalar(frame_idx: int) -> None:
        for _sid, net, env, _unit in lm_nets:
            g = float(env[min(max(frame_idx, 0), env.numel() - 1)])
            net.set_seq_gain(torch.tensor([g]), mode="prefix")

    orig_semantic = MiniMaxMusic3SemanticGenerationStep.__call__
    refresh_frames = max(1, int(args.refresh))
    window_frames = int(round(max(0.0, args.context_window) * frame_rate))

    def semantic_with_ramps(self, components, state):
        block_state = self.get_block_state(state)
        self.check_inputs(block_state)
        text_ids = block_state.text_ids
        max_frames = min(int(block_state.audio_duration * components.frame_rate), _MAX_AUDIO_FRAMES)
        if max_frames == 0:
            raise ValueError("audio_duration shorter than one frame")
        generator = block_state.generator
        language_model = components.language_model

        hooked = [
            model
            for model in (language_model, components.rvq_depth_decoder)
            if getattr(model, "_hf_hook", None) is not None
        ]
        for model in hooked:
            model._hf_hook.pre_forward(model)

        text_embeds = language_model.model.embed_tokens(text_ids)
        text_len = text_embeds.shape[1]

        # --dynamic-prompt: one caption variant per active pole, assembled
        # exactly like the pipeline's text step. Rebuilds pick the variant
        # matching the current gain, so the caption follows the fader.
        var_embeds: dict = {None: text_embeds}
        pos_base = text_len
        if args.dynamic_prompt:
            from diffusers.modular_pipelines.minimax_music3 import encoders as enc

            tokenizer = components.tokenizer
            for sid, _env in slider_envs:
                if sid not in POLE_OVERRIDES:
                    continue
                for sign, overrides in zip((-1, 1), POLE_OVERRIDES[sid]):
                    # Full caption with only the pole-owned sentences swapped;
                    # every other anchor (BPM, structure, continuity) stays.
                    text = (
                        f"{enc._IM_START}{enc._CAPTION_START}"
                        f"{enc._clean_caption(build_caption(overrides))}{enc._CAPTION_END}"
                        f"{enc._LYRICS_START}{enc._normalize_lyrics(LYRICS)}{enc._LYRICS_END}"
                        f"{enc._IM_END}{enc._AUDIO_START}"
                    )
                    ids = tokenizer(text, return_tensors="pt")["input_ids"]
                    unc = ids.clone()
                    unc[:, 1:-2] = enc._AUDIO_CFG_TOKEN_ID
                    ids = torch.cat((ids, unc), dim=0).to(text_ids.device)
                    var_embeds[(sid, sign)] = language_model.model.embed_tokens(ids)
            # Audio tokens always sit at pos_base + frame, whatever caption is
            # in front of them, so the audio timeline never shifts.
            pos_base = max(e.shape[1] for e in var_embeds.values())
            print(f"dynamic prompt: {len(var_embeds) - 1} pole variants, pos_base={pos_base}", flush=True)

        def pick_variant(frame_idx: int):
            if not args.dynamic_prompt:
                return None
            best, best_g = None, 0.0
            for sid, env in slider_envs:
                g = float(env[min(max(frame_idx, 0), env.numel() - 1)])
                if abs(g) > abs(best_g):
                    best, best_g = sid, g
            if best is None or abs(best_g) < args.prompt_gate:
                return None
            key = (best, 1 if best_g > 0 else -1)
            return key if key in var_embeds else None

        set_lm_scalar(0)
        output = language_model.model(inputs_embeds=text_embeds, use_cache=True)
        past_key_values = output.past_key_values
        last_hidden = output.last_hidden_state[:, -1]
        feedbacks: list[torch.Tensor] = []

        vocab_mask = torch.ones(language_model.config.vocab_size, dtype=torch.bool, device=text_ids.device)
        vocab_mask[_AUDIO_CODE_OFFSET : _AUDIO_CODE_OFFSET + _SEMANTIC_VOCAB_SIZE] = False
        vocab_mask[_AUDIO_END_TOKEN_ID] = False
        # Keep the take alive until the last half-second so the video has audio
        # under every HUD slot.
        end_guard = max(0, max_frames - int(0.5 * components.frame_rate))

        frame_hiddens = []
        for frame_index in range(max_frames + 1):
            emitted = max(0, frame_index - 1)
            set_lm_scalar(emitted)

            if frame_index > 0 and emitted > 0 and emitted % refresh_frames == 0 and feedbacks:
                # Re-encode prompt + recent history at the *current* gain:
                # the model believes the song has always been at g(t). At
                # constant gain this is the listen-set condition, so the
                # poles keep listen-set strength. History beyond the window
                # is dropped — the emitted tokens are literal audio evidence
                # (e.g. a male voice) and a long anchor outvotes any gain.
                kept = feedbacks if window_frames <= 0 else feedbacks[-window_frames:]
                first_kept = len(feedbacks) - len(kept)
                key = pick_variant(emitted)
                temb = var_embeds[key]
                full = torch.cat([temb, *kept], dim=1)
                # The kept window must stay at its true absolute positions.
                # If truncated history collapses onto the prompt, the model
                # believes it is forever ~3s into the song and never leaves
                # intro behavior: no verse progression, no singing.
                pos = (
                    torch.cat((torch.arange(temb.shape[1]), pos_base + first_kept + torch.arange(len(kept))))
                    .to(full.device)
                    .unsqueeze(0)
                    .expand(full.shape[0], -1)
                )
                set_lm_scalar(emitted)
                output = language_model.model(inputs_embeds=full, use_cache=True, position_ids=pos)
                past_key_values = output.past_key_values
                last_hidden = output.last_hidden_state[:, -1]
                active = ", ".join(
                    f"{sid}={float(env[min(emitted, env.numel() - 1)]):+.2f}"
                    for sid, _net, env, _unit in lm_nets
                    if abs(float(env[min(emitted, env.numel() - 1)])) > 1e-6
                )
                variant = f" caption={key[0]}{'+' if key[1] > 0 else '-'}" if key else ""
                print(
                    f"KV rebuild frame {emitted} ({emitted / components.frame_rate:.1f}s) "
                    f"{active or 'all-zero'}{variant}",
                    flush=True,
                )

            logits = language_model.lm_head(last_hidden).float()
            logits = logits.masked_fill(vocab_mask, -float("inf"))
            if emitted < end_guard:
                logits[..., _AUDIO_END_TOKEN_ID] = -float("inf")
            conditional, unconditional = logits[0:1], logits[1:2]
            guided = unconditional + (conditional - unconditional) * _AR_CFG_SCALE
            threshold = torch.topk(conditional, _AR_CFG_TOP_K, dim=-1).values[..., -1, None]
            guided = guided.masked_fill(conditional < threshold, -float("inf"))
            guided = guided.masked_fill(vocab_mask.unsqueeze(0), -float("inf"))
            sampled = _sample_top_k(guided, generator)
            if int(sampled.item()) == _AUDIO_END_TOKEN_ID:
                break

            semantic_code = sampled - _AUDIO_CODE_OFFSET
            frame_codes, depth_hidden = _generate_depth_codes(
                components, last_hidden, semantic_code.repeat(2), generator
            )
            if frame_index > 0:
                frame_hiddens.append(torch.cat((last_hidden[:1], depth_hidden), dim=-1))
                if len(frame_hiddens) >= max_frames:
                    break
            feedback = _embed_audio_frame(components, frame_codes)
            feedbacks.append(feedback)
            pos = torch.full(
                (feedback.shape[0], 1), pos_base + len(feedbacks) - 1, device=feedback.device, dtype=torch.long
            )
            output = language_model.model(
                inputs_embeds=feedback, past_key_values=past_key_values, use_cache=True, position_ids=pos
            )
            past_key_values = output.past_key_values
            last_hidden = output.last_hidden_state[:, -1]

        if not frame_hiddens:
            raise ValueError("generated zero audio frames")
        block_state.frame_hiddens = torch.stack(frame_hiddens, dim=1)
        print(f"AR frames={block_state.frame_hiddens.shape[1]}", flush=True)
        self.set_block_state(state, block_state)
        return components, state

    orig_cond = MiniMaxMusic3ChunkConditionStep.__call__

    def cond_with_slice(self, components, block_state, k):
        result = orig_cond(self, components, block_state, k)
        start = int(block_state.chunk_starts[k])
        n_frames = int(block_state.frame_hiddens.shape[1])
        end = min(start + 200, n_frames)
        # One gain per latent token, indexed by the latent's *global* AR
        # frame. A window that spans two slots gets the true step, not a
        # stretch-smeared slice. Position 0 is the prepended timestep token.
        latent_len = int(block_state.condition.shape[1])
        frames = start + torch.arange(latent_len) * (end - start) / max(latent_len, 1)
        idx = frames.round().long()
        for _sid, net, env, _unit in tf_nets:
            g = env[idx.clamp(min=0, max=env.numel() - 1)]
            # Timestep token takes the window's opening gain so a constant
            # envelope matches the listen-set uniform-multiplier condition.
            net.set_seq_gain(torch.cat((g[:1], g)), mode="prefix")
        return result

    MiniMaxMusic3SemanticGenerationStep.__call__ = semantic_with_ramps
    MiniMaxMusic3ChunkConditionStep.__call__ = cond_with_slice

    try:
        generator = torch.Generator(device).manual_seed(int(args.seed))
        with _apply(*apply_pairs):
            audio = pipe(
                prompt=NEUTRAL,
                lyrics=LYRICS,
                audio_duration=float(args.duration),
                generator=generator,
                output="audios",
            )[0]
    finally:
        MiniMaxMusic3SemanticGenerationStep.__call__ = orig_semantic
        MiniMaxMusic3ChunkConditionStep.__call__ = orig_cond

    duration, rms = _write_wav(out, audio, int(pipe.sampling_rate), float(args.duration) * 0.5)
    meta = {
        "method": "one_song_all_sliders",
        "duration": duration,
        "rms": rms,
        "seed": args.seed,
        "dynamic_prompt": bool(args.dynamic_prompt),
        "frame_rate": frame_rate,
        "max_frames": max_frames,
        "slot": SLOT,
        "schedule": schedule,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out} duration={duration:.2f}s rms={rms:.4f}", flush=True)


if __name__ == "__main__":
    main()
