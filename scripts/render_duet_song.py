#!/usr/bin/env python3
"""One song, a duet: the gender slider alternates male / female in sections
(the two never sing together), while the rhyme slider holds +1 the whole way
so both singers land tight AABB couplets.

Same machinery as render_song_ramp.py: LM halves get a per-frame scalar gain,
KV rebuilds every ~15 frames re-encode prompt + a short rolling context at the
*current* gain (with true absolute positions), and the caption swaps to the
active singer's pole variant at each rebuild. TF halves get per-latent gains
indexed by global AR frame.

  $PY scripts/render_duet_song.py \
      --out eval/listen/duet-60s/duet.wav --duration 60 --device 1
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

# Duet plan: sign of the gender slider per section, equal-length sections.
# -1 = male pole, +1 = female pole. Four 15s trades over a 60s take.
SECTION_SIGNS = [-1, 1, -1, 1]
XFADE = 1.5      # seconds to hand off between singers, passing through 0
SETTLE = 6.0     # seconds after the handoff where the incoming pole must win
OVERSHOOT = 1.4  # pole gain during settle; eases back to 1.0 after
SETTLE_WINDOW = 0.5   # rolling context (s) at rebuilds during settle — the
                      # outgoing singer's audio evidence must leave the window
                      # or it out-votes the flipped gain

RHYME_STYLE = (
    "Sung, not rapped, in tight AABB couplets where every line-ending rhymes and lands hard."
)

# Neutral caption states the duet contract; the pole variants swap only `who`
# while every other anchor (BPM, structure, continuity, rhyme style) stays.
CAP = {
    "genre": "Genre: pop. BPM 110.",
    "energy": "Medium energy, even mood.",
    "mix": "Dry-ish balanced studio mix.",
    "who": "A duet between one man and one woman who take turns singing lead and never "
           "sing at the same time. Whoever is singing is mixed upfront, no backing vocals, "
           "no harmonies.",
    "style": RHYME_STYLE,
    "cont": "A singer starts in the first bar and the lead vocal stays almost continuous "
            "to the end as the two trade sections.",
    "arr": "Verse-chorus pop. Drums, bass and guitar sit behind the vocal at a comfortable "
           "level. Vocals carry the song; instrumental gaps are short.",
}

POLE_WHO = {
    -1: "One male lead singer mixed upfront right now. A man is singing this section alone, "
        "his voice is deep and masculine. Even phrasing, no backing vocals.",
    1: "One female lead singer mixed upfront right now. A woman is singing this section alone, "
       "her voice is feminine. Even phrasing, no backing vocals.",
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

# AABB couplets throughout so the rhyme pole has something to bite on; verses
# trade perspective so the alternating voices read as two people answering
# each other rather than one singer changing costume.
LYRICS = (
    "[verse]\n"
    "I keep the low road, steady and slow\n"
    "I sing you the places you don't go\n"
    "My voice is the gravel under the wheel\n"
    "I tell you the half that I can feel\n"
    "[chorus]\n"
    "Trade it over, line for line\n"
    "Your verse is yours and mine is mine\n"
    "We never sing the same refrain\n"
    "But every answer rhymes again\n"
    "[verse]\n"
    "I take the high road over the hill\n"
    "I answer the verses you left still\n"
    "My voice is the silver over the stream\n"
    "I sing you the half that you can't dream\n"
    "[chorus]\n"
    "Trade it over, line for line\n"
    "Your verse is yours and mine is mine\n"
    "We never sing the same refrain\n"
    "But every answer rhymes again\n"
    "[verse]\n"
    "So pass me the melody, still warm\n"
    "I'll carry it safe through any storm\n"
    "I'll set it down at the top of the stair\n"
    "And trust you to find it waiting there\n"
    "[chorus]\n"
    "Trade it over, line for line\n"
    "Your verse is yours and mine is mine\n"
    "We never sing the same refrain\n"
    "But every answer rhymes again"
)


def _smooth(u: float) -> float:
    u = max(0.0, min(1.0, u))
    return u * u * (3 - 2 * u)


def duet_envelope(n: int, frame_rate: float, signs: list[int]) -> torch.Tensor:
    """Gender gain over AR frames: hold each section's pole, hand off through 0
    across XFADE seconds after each boundary, overshoot to OVERSHOOT for the
    settle period so the incoming voice punches through, then ease to 1."""
    env = torch.empty(n)
    section = n / len(signs)
    xf = max(1.0, XFADE * frame_rate)
    st = max(1.0, SETTLE * frame_rate)
    for i in range(n):
        s = min(int(i / section), len(signs) - 1)
        pos = i - s * section
        cur = float(signs[s])
        if s == 0:
            env[i] = cur
        elif pos < xf:
            prev = float(signs[s - 1])
            env[i] = prev + (cur * OVERSHOOT - prev) * _smooth(pos / xf)
        elif pos < xf + st:
            u = _smooth((pos - xf) / st)
            env[i] = cur * (OVERSHOOT + (1.0 - OVERSHOOT) * u)
        else:
            env[i] = cur
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument(
        "--refresh", type=int, default=15,
        help="AR frames between full-prefix KV rebuilds",
    )
    parser.add_argument(
        "--prompt-gate", type=float, default=0.35,
        help="|gender gain| above which that singer's caption replaces the duet one",
    )
    parser.add_argument(
        "--context-window", type=float, default=3.0,
        help="seconds of audio history kept at each KV rebuild; the voice cannot flip "
        "against a long anchored history",
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

    catalog = {s["id"]: s for s in registry.catalog()["sliders"]}
    for sid in ("gender", "rhyme"):
        if sid not in catalog:
            raise SystemExit(f"slider {sid!r} not in registry")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda:0"
    pipe = _load_pipeline(Path(args.model_dir), device)
    frame_rate = float(pipe.frame_rate)
    max_frames = int(args.duration * frame_rate)
    print(f"frame_rate={frame_rate} max_frames={max_frames}", flush=True)

    gender_env = duet_envelope(max_frames, frame_rate, SECTION_SIGNS)
    rhyme_env = torch.ones(max_frames)  # both singers rhyme, all song
    slider_envs = {"gender": gender_env, "rhyme": rhyme_env}

    lm_nets: list[tuple[str, object, torch.Tensor, float]] = []
    tf_nets: list[tuple[str, object, torch.Tensor, float]] = []
    apply_pairs = []
    for sid, env in slider_envs.items():
        comps = registry.resolve([{"id": sid, "scale": 1.0}])
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
    settle_window_frames = max(1, int(round(SETTLE_WINDOW * frame_rate)))

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

        # One caption variant per singer, assembled exactly like the pipeline's
        # text step. Rebuilds pick the variant matching the current gender gain.
        from diffusers.modular_pipelines.minimax_music3 import encoders as enc

        tokenizer = components.tokenizer
        var_embeds: dict = {None: text_embeds}
        for sign in (-1, 1):
            text = (
                f"{enc._IM_START}{enc._CAPTION_START}"
                f"{enc._clean_caption(build_caption({'who': POLE_WHO[sign]}))}{enc._CAPTION_END}"
                f"{enc._LYRICS_START}{enc._normalize_lyrics(LYRICS)}{enc._LYRICS_END}"
                f"{enc._IM_END}{enc._AUDIO_START}"
            )
            ids = tokenizer(text, return_tensors="pt")["input_ids"]
            unc = ids.clone()
            unc[:, 1:-2] = enc._AUDIO_CFG_TOKEN_ID
            ids = torch.cat((ids, unc), dim=0).to(text_ids.device)
            var_embeds[sign] = language_model.model.embed_tokens(ids)
        # Audio tokens always sit at pos_base + frame, whatever caption is in
        # front of them, so the audio timeline never shifts.
        pos_base = max(e.shape[1] for e in var_embeds.values())
        print(f"duet captions ready, pos_base={pos_base}", flush=True)

        def pick_variant(frame_idx: int):
            g = float(gender_env[min(max(frame_idx, 0), gender_env.numel() - 1)])
            if abs(g) < args.prompt_gate:
                return None
            return 1 if g > 0 else -1

        set_lm_scalar(0)
        output = language_model.model(inputs_embeds=text_embeds, use_cache=True)
        past_key_values = output.past_key_values
        last_hidden = output.last_hidden_state[:, -1]
        feedbacks: list[torch.Tensor] = []

        vocab_mask = torch.ones(language_model.config.vocab_size, dtype=torch.bool, device=text_ids.device)
        vocab_mask[_AUDIO_CODE_OFFSET : _AUDIO_CODE_OFFSET + _SEMANTIC_VOCAB_SIZE] = False
        vocab_mask[_AUDIO_END_TOKEN_ID] = False
        end_guard = max(0, max_frames - int(0.5 * components.frame_rate))

        frame_hiddens = []
        for frame_index in range(max_frames + 1):
            emitted = max(0, frame_index - 1)
            set_lm_scalar(emitted)

            if frame_index > 0 and emitted > 0 and emitted % refresh_frames == 0 and feedbacks:
                # Re-encode prompt + recent history at the *current* gain; a
                # short window lets the voice actually flip at each handoff.
                # During settle after a boundary the window shrinks further so
                # the outgoing singer's evidence cannot re-anchor the voice.
                sec = max_frames / len(SECTION_SIGNS)
                pos = emitted - int(emitted // sec) * sec
                in_settle = emitted >= sec and pos < (XFADE + SETTLE) * components.frame_rate
                w = settle_window_frames if in_settle else window_frames
                kept = feedbacks if w <= 0 else feedbacks[-w:]
                first_kept = len(feedbacks) - len(kept)
                key = pick_variant(emitted)
                temb = var_embeds[key]
                full = torch.cat([temb, *kept], dim=1)
                # Kept window stays at its true absolute positions.
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
                g = float(gender_env[min(emitted, gender_env.numel() - 1)])
                who = {None: "handoff", -1: "male", 1: "female"}[key]
                print(
                    f"KV rebuild frame {emitted} ({emitted / components.frame_rate:.1f}s) "
                    f"gender={g:+.2f} rhyme=+1.00 caption={who} window={len(kept)}fr",
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
        latent_len = int(block_state.condition.shape[1])
        frames = start + torch.arange(latent_len) * (end - start) / max(latent_len, 1)
        idx = frames.round().long()
        for _sid, net, env, _unit in tf_nets:
            g = env[idx.clamp(min=0, max=env.numel() - 1)]
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
    section = float(args.duration) / len(SECTION_SIGNS)
    meta = {
        "method": "duet_gender_alternate_rhyme_flat",
        "duration": duration,
        "rms": rms,
        "seed": args.seed,
        "frame_rate": frame_rate,
        "max_frames": max_frames,
        "xfade": XFADE,
        "settle": SETTLE,
        "overshoot": OVERSHOOT,
        "settle_window": SETTLE_WINDOW,
        "sections": [
            {"t0": i * section, "t1": (i + 1) * section, "singer": "male" if s < 0 else "female"}
            for i, s in enumerate(SECTION_SIGNS)
        ],
        "rhyme": 1.0,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out} duration={duration:.2f}s rms={rms:.4f}", flush=True)


if __name__ == "__main__":
    main()
