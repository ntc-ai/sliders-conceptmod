#!/usr/bin/env python3
"""Line-by-line duet: the gender slider flips per lyric *line*, driven by the
LM's own attention. There is no ground-truth lyric clock during AR generation,
but each new audio token's attention over the prompt concentrates on the lyric
tokens it is currently singing — so every few frames we read that distribution,
score it per line, and advance a monotonic "current line" estimate. Even lines
are sung male (gain -OVERSHOOT), odd lines female (+OVERSHOOT), rhyme stays +1.

The realized per-frame gain is recorded and reused for the TF decode, and the
full alignment trace (time -> line) is written to the output .json so a page
can highlight the current line while the audio plays.

Same core machinery as render_duet_song.py: current-gain KV rebuilds with true
absolute positions and a short rolling context, per-singer caption variants,
global-frame TF gains.

  $PY scripts/render_duet_lines.py \
      --out eval/listen/duet-60s/duet-lines.wav --duration 60 --device 1
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

OVERSHOOT = 1.4       # pole gain while a flip settles (v3 finding)
SLEW_SECONDS = 0.8    # full -OVERSHOOT -> +OVERSHOOT swing time
WINDOW_SECONDS = 3.0  # normal rolling context; a permanently short window
                      # erases lyric progress and the singer loops line one
SETTLE_SECONDS = 3.5  # after a detected line flip: short window + overshoot
SETTLE_WINDOW = 0.5   # context during settle so the old voice drains fast
ALIGN_EVERY = 5       # AR frames between attention reads (0.2s at 25 fps)
# Lyric-alignment heads (layer, head), found by --diagnose on this model:
# expected lyric position of these heads tracks time with corr 0.89-0.94.
ALIGN_HEADS = [(29, 22), (31, 18), (29, 21), (31, 8), (29, 20)]
START_HOLD_SECONDS = 4.0  # no tracker updates during the instrumental intro —
                          # its attention is noise and racing there is
                          # amplified by the lyric-consuming prompt
MIN_LINE_SECONDS = 3.0    # a line takes ~5s to sing; it cannot be left sooner
                          # than this, which caps the tracker's advance rate
KEEP_BEHIND = 1           # sung lines kept in the prompt behind the current
                          # one, so a small overshoot stays recoverable

RHYME_STYLE = (
    "Sung, not rapped, in tight AABB couplets where every line-ending rhymes and lands hard."
)

CAP = {
    "genre": "Genre: pop. BPM 110.",
    "energy": "Medium energy, even mood.",
    "mix": "Dry-ish balanced studio mix.",
    "who": "A duet between one man and one woman who trade the lead line by line and never "
           "sing at the same time. Whoever is singing is mixed upfront, no backing vocals, "
           "no harmonies.",
    "style": RHYME_STYLE,
    "cont": "A singer starts in the first bar and the lead vocal stays almost continuous "
            "to the end as the two trade lines.",
    "arr": "Verse-chorus pop. Drums, bass and guitar sit behind the vocal at a comfortable "
           "level. Vocals carry the song; instrumental gaps are short.",
}

POLE_WHO = {
    -1: "One male lead singer mixed upfront right now. A man is singing this line alone, "
        "his voice is deep and masculine. Even phrasing, no backing vocals.",
    1: "One female lead singer mixed upfront right now. A woman is singing this line alone, "
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

# Every line is textually unique: repeated chorus lines are indistinguishable
# to the alignment heads (the tracker cannot tell which chorus is being sung),
# so the song is through-composed call-and-response in AABB couplets.
LYRICS = (
    "[verse]\n"
    "I keep the low road, steady and slow\n"
    "I sing you the places you don't go\n"
    "My voice is the gravel under the wheel\n"
    "I tell you the half that I can feel\n"
    "[verse]\n"
    "I take the high road over the hill\n"
    "I answer the verses you left still\n"
    "My voice is the silver on the stream\n"
    "I sing you the half that you can't dream\n"
    "[verse]\n"
    "Pass me the melody, keep it warm\n"
    "I'll carry it safe through any storm\n"
    "Set it down at the top of the stair\n"
    "I'll find it waiting anywhere\n"
    "[verse]\n"
    "You hold the morning, I'll hold the night\n"
    "You take the shadow, I'll take the light\n"
    "Your line is heavy and mine is thin\n"
    "You end the verse and I begin\n"
    "[verse]\n"
    "I lay the rhythm brick by brick\n"
    "I make the slow hours move too quick\n"
    "I plant the question in the ground\n"
    "I turn your silence into sound\n"
    "[verse]\n"
    "We trade the lead but never meet\n"
    "Two different voices, one heartbeat\n"
    "You sang the door and I sang the key\n"
    "The last line's yours, the first was me"
)


def sung_line_spans(lyrics_norm: str) -> list[tuple[int, int, str]]:
    """(char_start, char_end, text) per sung line; a tag line ([verse] etc.)
    is folded into the following sung line's span so every lyric token maps
    to some sung line."""
    spans = []
    pos = 0
    pending_start = None
    for raw in lyrics_norm.split("\n"):
        start, end = pos, pos + len(raw)
        pos = end + 1
        if raw.strip().startswith("["):
            if pending_start is None:
                pending_start = start
            continue
        s = pending_start if pending_start is not None else start
        pending_start = None
        spans.append((s, end, raw.strip()))
    return spans


def token_line_map(tokenizer, full_text: str, lyrics_norm: str, n_lines_spans) -> torch.Tensor:
    """line id per prompt token (-1 = not lyrics). Incremental prefix
    tokenization; off-by-one at merge boundaries is fine at this granularity."""
    lyr_start = full_text.index(lyrics_norm)
    n_total = len(tokenizer(full_text)["input_ids"])
    tok_line = torch.full((n_total,), -1, dtype=torch.long)
    prev = len(tokenizer(full_text[:lyr_start])["input_ids"])
    for li, (_s, end, _t) in enumerate(n_lines_spans):
        t_end = len(tokenizer(full_text[: lyr_start + end])["input_ids"])
        tok_line[prev : min(t_end, n_total)] = li
        prev = min(t_end, n_total)
    return tok_line


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--refresh", type=int, default=15)
    parser.add_argument("--prompt-gate", type=float, default=0.35)
    parser.add_argument(
        "--diagnose", action="store_true",
        help="render neutral (gender 0, no caption swap, no line advance) and dump every "
        "head's expected lyric position over time, to find the alignment heads",
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

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda:0"
    pipe = _load_pipeline(Path(args.model_dir), device)
    frame_rate = float(pipe.frame_rate)
    max_frames = int(args.duration * frame_rate)
    print(f"frame_rate={frame_rate} max_frames={max_frames}", flush=True)

    lm_nets, tf_nets, apply_pairs = [], [], []
    for sid in ("gender", "rhyme"):
        for comp in registry.resolve([{"id": sid, "scale": 1.0}]):
            net, _meta = _wrap_sidecar(pipe, device, Path(comp["weights"]), comp["kind"])
            unit = float(comp["multiplier"])
            net.set_lora_slider(unit)
            apply_pairs.append((net, unit))
            (lm_nets if comp["kind"] == "language_model" else tf_nets).append((sid, net, unit))
            print(f"{sid:8} {comp['kind']:16} {Path(comp['weights']).name} unit={unit:+.3f}", flush=True)

    # Realized per-frame gender gain, filled during AR, read by the TF decode.
    realized = {"env": None}
    trace: list[dict] = []

    orig_semantic = MiniMaxMusic3SemanticGenerationStep.__call__
    refresh_frames = max(1, int(args.refresh))
    window_frames = max(1, int(round(WINDOW_SECONDS * frame_rate)))
    settle_window_frames = max(1, int(round(SETTLE_WINDOW * frame_rate)))
    slew_per_frame = 2.0 * OVERSHOOT / max(1.0, SLEW_SECONDS * frame_rate)

    def semantic_with_lines(self, components, state):
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

        from diffusers.modular_pipelines.minimax_music3 import encoders as enc

        tokenizer = components.tokenizer
        lyrics_norm = enc._normalize_lyrics(LYRICS)
        spans = sung_line_spans(lyrics_norm)
        n_lines = len(spans)
        line_sign = [(-1 if i % 2 == 0 else 1) for i in range(n_lines)]
        print(f"{n_lines} sung lines, even=male odd=female", flush=True)

        def full_text_for(caption: str) -> str:
            return (
                f"{enc._IM_START}{enc._CAPTION_START}"
                f"{enc._clean_caption(caption)}{enc._CAPTION_END}"
                f"{enc._LYRICS_START}{lyrics_norm}{enc._LYRICS_END}"
                f"{enc._IM_END}{enc._AUDIO_START}"
            )

        # Prompt variants are built lazily per (caption sign, first remaining
        # line): at each rebuild the lyric block starts at the tracked current
        # line, so already-sung lines physically leave the prompt and the model
        # cannot loop them — the rolling window alone cannot carry lyric
        # progress. Each variant carries its own token->line map (ids stay
        # absolute) for the alignment read.
        emb_cache: dict = {}

        def variant_embeds(sign, start_line: int):
            key = (sign, start_line)
            if key in emb_cache:
                return emb_cache[key]
            rem_start = spans[start_line][0] if start_line > 0 else 0
            rem = lyrics_norm[rem_start:]
            caption = NEUTRAL if sign is None else build_caption({"who": POLE_WHO[sign]})
            text = (
                f"{enc._IM_START}{enc._CAPTION_START}"
                f"{enc._clean_caption(caption)}{enc._CAPTION_END}"
                f"{enc._LYRICS_START}{rem}{enc._LYRICS_END}"
                f"{enc._IM_END}{enc._AUDIO_START}"
            )
            ids = tokenizer(text, return_tensors="pt")["input_ids"]
            unc = ids.clone()
            unc[:, 1:-2] = enc._AUDIO_CFG_TOKEN_ID
            ids = torch.cat((ids, unc), dim=0).to(text_ids.device)
            emb = language_model.model.embed_tokens(ids)
            lyr_start = text.index(enc._LYRICS_START) + len(enc._LYRICS_START)
            n_total = ids.shape[1]
            tok_line = torch.full((n_total,), -1, dtype=torch.long)
            prev = len(tokenizer(text[:lyr_start])["input_ids"])
            for li in range(start_line, n_lines):
                end = spans[li][1] - rem_start
                t_end = min(len(tokenizer(text[: lyr_start + end])["input_ids"]), n_total)
                tok_line[prev:t_end] = li
                prev = t_end
            emb_cache[key] = (emb, tok_line)
            return emb_cache[key]

        pos_base = max(variant_embeds(s, 0)[0].shape[1] for s in (None, -1, 1))
        print(f"pos_base={pos_base}, align heads={ALIGN_HEADS}", flush=True)

        # Mutable alignment / gain state driven inside the AR loop.
        st = {
            "line": 0,
            "gain": 0.0 if args.diagnose else float(line_sign[0]) * OVERSHOOT,
            "tok_map": variant_embeds(None, 0)[1],
            "settle_until": int(SETTLE_SECONDS * components.frame_rate),
            "line_since": 0.0,
        }
        diag = {"t": [], "exp": [], "mass": []}

        def diag_sample(attentions) -> None:
            """Per (layer, head): expected lyric line of this token's attention,
            and the fraction of its attention mass that lands on lyric tokens."""
            tok_line = st["tok_map"]
            k = min(attentions[0].shape[-1], tok_line.numel())
            valid = tok_line[:k] >= 0
            ids = tok_line[:k][valid].float()
            exps, masses = [], []
            for a in attentions:
                row = a[0, :, -1, :].float().cpu()  # [H, K]
                w = row[:, :k][:, valid]
                exps.append((w * ids).sum(-1) / (w.sum(-1) + 1e-9))
                masses.append(w.sum(-1) / (row.sum(-1) + 1e-9))
            diag["exp"].append(torch.stack(exps))
            diag["mass"].append(torch.stack(masses))
            diag["t"].append(len(feedbacks) / components.frame_rate)

        def line_scores(attentions) -> torch.Tensor:
            """Attention mass per sung line from the known alignment heads
            (per-head renormalized over lyric tokens)."""
            tok_line = st["tok_map"]
            k = min(attentions[0].shape[-1], tok_line.numel())
            valid = tok_line[:k] >= 0
            idx = tok_line[:k][valid]
            scores = torch.zeros(n_lines)
            for l, h in ALIGN_HEADS:
                w = attentions[l][0, h, -1, :k].float().cpu()[valid]
                scores.index_add_(0, idx, w / (w.sum() + 1e-9))
            return scores

        # HMM filter over lines: singing moves forward slowly, the intro is
        # attention noise, and a wrong step must be recoverable — a hard
        # monotonic argmax raced ahead during the intro and then trapped
        # itself. Transition prior: mostly stay, sometimes +1, rarely +2,
        # tiny -1 for recovery.
        post = torch.zeros(n_lines)
        post[0], post[1] = 0.8, 0.2
        TRANS = ((0, 0.945), (1, 0.04), (2, 0.005), (-1, 0.01))

        def hmm_update(scores: torch.Tensor) -> tuple[int, float]:
            obs = scores / (scores.sum() + 1e-9)
            prior = torch.zeros_like(post)
            for d, p in TRANS:
                if d == 0:
                    prior += p * post
                elif d > 0:
                    prior[d:] += p * post[: n_lines - d]
                else:
                    prior[:-1] += p * post[1:]
            new = prior * (obs + 1e-6)
            post.copy_(new / (new.sum() + 1e-9))
            return int(torch.argmax(post)), float(post.max())

        def set_lm_gains() -> None:
            for sid, net, _unit in lm_nets:
                g = st["gain"] if sid == "gender" else 1.0
                net.set_seq_gain(torch.tensor([g]), mode="prefix")

        env = torch.zeros(max_frames)

        set_lm_gains()
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

            # Slew the gain toward the current line's pole (overshot while the
            # flip settles) and record it.
            settling = emitted < st["settle_until"]
            target = 0.0 if args.diagnose else float(line_sign[st["line"]]) * (OVERSHOOT if settling else 1.0)
            delta = max(-slew_per_frame, min(slew_per_frame, target - st["gain"]))
            st["gain"] += delta
            if emitted < max_frames:
                env[emitted] = st["gain"]
            set_lm_gains()

            if frame_index > 0 and emitted > 0 and emitted % refresh_frames == 0 and feedbacks:
                kept = feedbacks[-(settle_window_frames if settling else window_frames):]
                first_kept = len(feedbacks) - len(kept)
                key = None
                if abs(st["gain"]) >= args.prompt_gate:
                    key = 1 if st["gain"] > 0 else -1
                temb, st["tok_map"] = variant_embeds(key, max(0, st["line"] - KEEP_BEHIND))
                full = torch.cat([temb, *kept], dim=1)
                pos = (
                    torch.cat((torch.arange(temb.shape[1]), pos_base + first_kept + torch.arange(len(kept))))
                    .to(full.device)
                    .unsqueeze(0)
                    .expand(full.shape[0], -1)
                )
                set_lm_gains()
                output = language_model.model(inputs_embeds=full, use_cache=True, position_ids=pos)
                past_key_values = output.past_key_values
                last_hidden = output.last_hidden_state[:, -1]

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
            want_align = len(feedbacks) % ALIGN_EVERY == 0
            # SDPA never materializes attention weights; flip this one-token
            # forward to eager (dispatch reads config at call time). q_len=1,
            # so the eager pass is cheap.
            attn_impl = language_model.config._attn_implementation
            if want_align:
                language_model.config._attn_implementation = "eager"
            try:
                output = language_model.model(
                    inputs_embeds=feedback,
                    past_key_values=past_key_values,
                    use_cache=True,
                    position_ids=pos,
                    output_attentions=want_align,
                )
            finally:
                language_model.config._attn_implementation = attn_impl
            past_key_values = output.past_key_values
            last_hidden = output.last_hidden_state[:, -1]

            if want_align and output.attentions and args.diagnose:
                diag_sample(output.attentions)
            elif want_align and output.attentions:
                t_now = len(feedbacks) / components.frame_rate
                dwell_ok = (
                    t_now >= START_HOLD_SECONDS
                    and t_now - st["line_since"] >= MIN_LINE_SECONDS
                )
                if dwell_ok:
                    est, conf = hmm_update(line_scores(output.attentions))
                else:
                    est, conf = st["line"], 0.0
                if est != st["line"]:
                    st["line_since"] = t_now
                    if line_sign[est] != line_sign[st["line"]]:
                        st["settle_until"] = len(feedbacks) + int(SETTLE_SECONDS * components.frame_rate)
                st["line"] = est
                t = len(feedbacks) / components.frame_rate
                trace.append(
                    {
                        "t": round(t, 2),
                        "line": st["line"],
                        "est": est,
                        "conf": round(conf, 3),
                        "gain": round(st["gain"], 2),
                    }
                )
                if len(trace) % 10 == 1:
                    print(
                        f"align t={t:5.1f}s line={st['line']:2d} "
                        f"({spans[st['line']][2][:32]!r}) conf={conf:.2f} gain={st['gain']:+.2f}",
                        flush=True,
                    )

        if not frame_hiddens:
            raise ValueError("generated zero audio frames")
        if args.diagnose and diag["exp"]:
            torch.save(
                {
                    "t": torch.tensor(diag["t"]),
                    "exp": torch.stack(diag["exp"]),    # [T, layers, heads]
                    "mass": torch.stack(diag["mass"]),  # [T, layers, heads]
                    "n_lines": n_lines,
                },
                out.with_suffix(".diag.pt"),
            )
            print(f"wrote {out.with_suffix('.diag.pt')} ({len(diag['t'])} samples)", flush=True)
        realized["env"] = env[: len(frame_hiddens)]
        realized["spans"] = spans
        realized["line_sign"] = line_sign
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
        genv = realized["env"]
        for sid, net, _unit in tf_nets:
            env = genv if sid == "gender" else torch.ones_like(genv)
            g = env[idx.clamp(min=0, max=env.numel() - 1)]
            net.set_seq_gain(torch.cat((g[:1], g)), mode="prefix")
        return result

    MiniMaxMusic3SemanticGenerationStep.__call__ = semantic_with_lines
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
        "method": "duet_line_by_line_attention_aligned",
        "duration": duration,
        "rms": rms,
        "seed": args.seed,
        "frame_rate": frame_rate,
        "overshoot": OVERSHOOT,
        "slew_seconds": SLEW_SECONDS,
        "window_seconds": WINDOW_SECONDS,
        "lines": [
            {"i": i, "text": t, "singer": "male" if s < 0 else "female"}
            for i, ((_a, _b, t), s) in enumerate(zip(realized["spans"], realized["line_sign"]))
        ],
        "trace": trace,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out} duration={duration:.2f}s rms={rms:.4f} trace={len(trace)} samples", flush=True)


if __name__ == "__main__":
    main()
