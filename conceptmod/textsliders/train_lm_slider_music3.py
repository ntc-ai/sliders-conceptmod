"""Notrigger-style slider on MiniMax Music 3's language model.

Vocal identity is chosen in the AR stage, not the flow transformer. This trains
LoRA on Qwen3Attention so a neutral caption + LoRA scale moves the prompt
hidden state toward a female or male caption.

Audio-end regularizer: a cut ends when the AR language model samples
<|audio_end|>; LM LoRAs perturb those logits, and un-regularized sliders
measurably suppress the end token, so stacked sliders blow through the
duration cap and get truncated mid-phrase. With --endreg_weight > 0 each
prompt row is pre-rolled once with the base model (the same CFG compose loop
inference uses; cached on disk), and every training step teacher-forces the
LoRA'd model over that composition, penalizing drift of the end margin —
logit(<|audio_end|>) minus logsumexp over the semantic-code band — at every
decode position. The slider keeps moving the musical plan; it must not move
the stop-decision axis.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HF_HOME = "/ml2/music/.cache/huggingface"
os.environ["HF_HOME"] = _HF_HOME
os.environ["HUGGINGFACE_HUB_CACHE"] = f"{_HF_HOME}/hub"
os.environ["HF_HUB_CACHE"] = f"{_HF_HOME}/hub"
os.environ["TRANSFORMERS_CACHE"] = f"{_HF_HOME}/hub"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conceptmod.textsliders.lora import LoRANetwork

DEFAULT_MODEL = Path("/ml2/music/models/MiniMax-Music3")
TARGET_REPLACE = ["Qwen3Attention"]

# Row fields this trainer actually consumes (`attributes` is expanded away by
# _expand_attributes). Anything else in the YAML - `action`, `guidance_scale`,
# `batch_size`, `unconditional` - is only read by the transformer trainer.
_USED_ROW_KEYS = frozenset({"target", "positive", "negative", "neutral", "lyrics", "attributes"})

_IM_START, _IM_END = "<|im_start|>", "<|im_end|>"
_CAPTION_START, _CAPTION_END = "<|caption_start|>", "<|caption_end|>"
_LYRICS_START, _LYRICS_END = "<|lyrics_start|>", "<|lyrics_end|>"
_AUDIO_START = "<|audio_start|>"


def _assemble(prompt: str, lyrics: str) -> str:
    from diffusers.modular_pipelines.minimax_music3.encoders import (
        _clean_caption,
        _normalize_lyrics,
    )

    return (
        f"{_IM_START}{_CAPTION_START}{_clean_caption(prompt)}{_CAPTION_END}"
        f"{_LYRICS_START}{_normalize_lyrics(lyrics)}{_LYRICS_END}{_IM_END}{_AUDIO_START}"
    )


def _load_rows(path: Path) -> tuple[list[dict], dict]:
    """All prompt rows (with `attributes` expansion) plus top-level label metadata."""
    from conceptmod.textsliders.train_lora_music3 import _expand_attributes

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    meta: dict = {}
    if isinstance(raw, dict):
        meta = {
            "plus_label": str(raw.get("plus_label") or ""),
            "minus_label": str(raw.get("minus_label") or ""),
            "recommended_range": raw.get("recommended_range") or [-2.0, 2.0],
        }
        raw = raw.get("rows")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"prompts file is empty: {path}")
    ignored = sorted(
        {
            str(key)
            for item in raw
            if isinstance(item, dict)
            for key in item
            if str(key) not in _USED_ROW_KEYS
        }
    )
    if ignored:
        print(f"note: ignoring per-row fields not used by the LM trainer: {', '.join(ignored)}")
    rows: list[dict] = []
    for item in raw:
        rows.extend(_expand_attributes(item))
    return rows, meta


def _tokenize(tokenizer, text: str, device: torch.device):
    out = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return out["input_ids"].to(device), out["attention_mask"].to(device)


@torch.no_grad()
def _encode_static(lm, input_ids, attention_mask) -> torch.Tensor:
    hidden = lm.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    # Last real token (the audio-start token) is what AR continues from.
    lengths = attention_mask.sum(dim=1) - 1
    gather = lengths.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
    last = hidden.gather(1, gather.clamp(min=0)).squeeze(1)
    return last.float()


def _encode_train(lm, input_ids, attention_mask) -> torch.Tensor:
    hidden = lm.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    lengths = attention_mask.sum(dim=1) - 1
    gather = lengths.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
    last = hidden.gather(1, gather.clamp(min=0)).squeeze(1)
    return last.float()


def _set_scale(network: LoRANetwork, scale: float) -> None:
    network.set_lora_slider(scale)
    for lora in network.unet_loras:
        lora.multiplier = float(scale)


def _frame_margins(lm, hidden_tail: torch.Tensor) -> torch.Tensor:
    """End margin — logit(<|audio_end|>) - logsumexp(semantic-code band) — at
    each decode position. hidden_tail: [1, P, hidden]; returns float32 [P].
    Only the lm_head rows that decide stop-vs-continue are multiplied out."""
    from diffusers.modular_pipelines.minimax_music3.encoders import (
        _AUDIO_CODE_OFFSET,
        _AUDIO_END_TOKEN_ID,
        _SEMANTIC_VOCAB_SIZE,
    )

    weight = lm.lm_head.weight
    hidden = hidden_tail[0]
    end_logit = (hidden @ weight[_AUDIO_END_TOKEN_ID]).float()
    sem_logits = (hidden @ weight[_AUDIO_CODE_OFFSET : _AUDIO_CODE_OFFSET + _SEMANTIC_VOCAB_SIZE].T).float()
    return end_logit - torch.logsumexp(sem_logits, dim=-1)


def _forward_teacher_forced(lm, prompt_embeds: torch.Tensor, frame_embeds: torch.Tensor | None):
    """One forward over [prompt; composed frames]. Returns the prompt-last
    hidden state (the slider target position — causality makes it identical to
    a prompt-only forward), the end margins at positions prompt-last..end, and
    the full hidden sequence for optional plan-drift regularization."""
    embeds = prompt_embeds if frame_embeds is None else torch.cat((prompt_embeds, frame_embeds), dim=1)
    mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device)
    hidden = lm.model(inputs_embeds=embeds, attention_mask=mask).last_hidden_state
    prompt_last = hidden[:, prompt_embeds.shape[1] - 1]
    margins = _frame_margins(lm, hidden[:, prompt_embeds.shape[1] - 1 :])
    return prompt_last.float(), margins, hidden


@torch.no_grad()
def _preroll_frames(lm, depth_decoder, cond_ids: torch.Tensor, frames_cap: int, seed: int, device):
    """Compose up to frames_cap frames with the base model, exactly like
    inference (CFG pair, top-k, depth codes, feedback embeddings). Returns
    (frame_embeds [1, N, hidden] or None, ended_naturally)."""
    from types import SimpleNamespace

    from diffusers.modular_pipelines.minimax_music3.encoders import (
        _AR_CFG_SCALE,
        _AR_CFG_TOP_K,
        _AUDIO_CFG_TOKEN_ID,
        _AUDIO_CODE_OFFSET,
        _AUDIO_END_TOKEN_ID,
        _SEMANTIC_VOCAB_SIZE,
        _embed_audio_frame,
        _generate_depth_codes,
        _sample_top_k,
    )

    shim = SimpleNamespace(
        language_model=lm,
        rvq_depth_decoder=depth_decoder,
        num_codebooks=int(depth_decoder.config.num_codebooks),
        audio_vocab_size=int(depth_decoder.config.audio_vocab_size),
    )
    unconditional_ids = cond_ids.clone()
    unconditional_ids[:, 1:-2] = _AUDIO_CFG_TOKEN_ID
    text_ids = torch.cat((cond_ids, unconditional_ids), dim=0)
    generator = torch.Generator(device).manual_seed(seed)

    output = lm.model(inputs_embeds=lm.model.embed_tokens(text_ids), use_cache=True)
    past_key_values = output.past_key_values
    last_hidden = output.last_hidden_state[:, -1]

    vocab_mask = torch.ones(lm.config.vocab_size, dtype=torch.bool, device=device)
    vocab_mask[_AUDIO_CODE_OFFSET : _AUDIO_CODE_OFFSET + _SEMANTIC_VOCAB_SIZE] = False
    vocab_mask[_AUDIO_END_TOKEN_ID] = False

    frame_embeds: list[torch.Tensor] = []
    ended = False
    while len(frame_embeds) < frames_cap:
        logits = lm.lm_head(last_hidden).float()
        logits = logits.masked_fill(vocab_mask, -float("inf"))
        conditional, unconditional = logits[0:1], logits[1:2]
        guided = unconditional + (conditional - unconditional) * _AR_CFG_SCALE
        threshold = torch.topk(conditional, _AR_CFG_TOP_K, dim=-1).values[..., -1, None]
        guided = guided.masked_fill(conditional < threshold, -float("inf"))
        guided = guided.masked_fill(vocab_mask.unsqueeze(0), -float("inf"))
        sampled = _sample_top_k(guided, generator)
        if int(sampled.item()) == _AUDIO_END_TOKEN_ID:
            ended = True
            break
        semantic_code = sampled - _AUDIO_CODE_OFFSET
        frame_codes, _depth_hidden = _generate_depth_codes(shim, last_hidden, semantic_code.repeat(2), generator)
        feedback = _embed_audio_frame(shim, frame_codes)
        frame_embeds.append(feedback[0:1])
        output = lm.model(inputs_embeds=feedback, past_key_values=past_key_values, use_cache=True)
        past_key_values = output.past_key_values
        last_hidden = output.last_hidden_state[:, -1]

    frames = torch.cat(frame_embeds, dim=1) if frame_embeds else None
    return frames, ended


def _endreg_cache_path(cache_dir: Path, model_dir: str, text: str, frames_cap: int, seed: int) -> Path:
    import hashlib

    digest = hashlib.sha256(
        json.dumps([str(model_dir), text, int(frames_cap), int(seed)]).encode("utf-8")
    ).hexdigest()[:16]
    return cache_dir / f"preroll-{digest}.pt"


def train(args: argparse.Namespace) -> Path:
    device = torch.device(f"cuda:{int(args.device)}")
    # Pin every consumer of global RNG (LoRA init is the only one) so two runs
    # differing only in --seed are step-for-step comparable, as in the
    # transformer trainer's determinism fix. The endreg pre-roll carries its
    # own generator (--endreg_seed).
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    import random

    random.seed(int(args.seed))
    rows, prompts_meta = _load_rows(Path(args.prompts_file))
    if args.plus_label:
        prompts_meta["plus_label"] = args.plus_label
    if args.minus_label:
        prompts_meta["minus_label"] = args.minus_label

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(args.model_dir) / "tokenizer"),
        local_files_only=True,
    )
    lm = AutoModelForCausalLM.from_pretrained(
        str(Path(args.model_dir) / "language_model"),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    lm.to(device)
    lm.eval()
    lm.requires_grad_(False)

    beta = float(args.common_beta)
    row_data = []
    for index, row in enumerate(rows):
        lyrics = str(row["lyrics"])
        texts = {
            "neutral": _assemble(str(row.get("neutral") or row["target"]), lyrics),
            "positive": _assemble(str(row["positive"]), lyrics),
            "negative": _assemble(str(row["negative"]), lyrics),
        }
        tokens = {name: _tokenize(tokenizer, text, device) for name, text in texts.items()}
        with torch.no_grad():
            pos_tgt = _encode_static(lm, *tokens["positive"])
            neg_tgt = _encode_static(lm, *tokens["negative"])
            neu_ref = _encode_static(lm, *tokens["neutral"])
        target_cos = F.cosine_similarity(pos_tgt - neu_ref, neg_tgt - neu_ref, dim=-1).mean().item()
        print(
            f"row {index}: tokens={tokens['neutral'][0].shape[1]} "
            f"L2 pos={torch.norm(pos_tgt - neu_ref).item():.3f} "
            f"neg={torch.norm(neg_tgt - neu_ref).item():.3f} "
            f"cos(pos-neu, neg-neu)={target_cos:.3f}"
            f"{'  <- collapse inherited from raw targets' if target_cos > 0.3 else ''}"
        )
        if args.symmetric:
            # Raw pos/neg targets share a large common component vs neutral (both
            # captions add axis-independent detail); training against them makes
            # +1 and -1 point the same way. Antisymmetrize around neutral instead.
            axis = (pos_tgt - neg_tgt) / 2.0 * float(args.target_scale)
            common = (pos_tgt + neg_tgt) / 2.0 - neu_ref
            tgt_plus = neu_ref + axis + beta * common
            tgt_minus = neu_ref - axis + beta * common
        else:
            if float(args.target_scale) != 1.0:
                raise ValueError("--target_scale is defined for symmetric targets only")
            tgt_plus, tgt_minus = pos_tgt, neg_tgt
        row_data.append(
            {
                "tokens": tokens["neutral"],
                "text_neutral": texts["neutral"],
                "tgt_plus": tgt_plus,
                "tgt_minus": tgt_minus,
                "neu_ref": neu_ref,
            }
        )

    endreg = args.endreg_weight > 0 and args.endreg_frames > 0
    if endreg:
        # Pre-roll every row with the pristine base model (the LoRA does not
        # exist yet), teacher-force the same sequence once, and freeze the base
        # end margins the slider must preserve. The compose is the expensive
        # part, so it is cached on disk; margins are recomputed per run.
        cache_dir = Path(args.endreg_cache)
        cache_dir.mkdir(parents=True, exist_ok=True)
        depth_decoder = None
        ended_count = 0
        for index, data in enumerate(row_data):
            cache_path = _endreg_cache_path(
                cache_dir, args.model_dir, data["text_neutral"], args.endreg_frames, args.endreg_seed + index
            )
            if cache_path.exists():
                blob = torch.load(cache_path, map_location=device, weights_only=True)
            else:
                if depth_decoder is None:
                    from diffusers import MiniMaxMusic3RVQDepthDecoder

                    depth_decoder = MiniMaxMusic3RVQDepthDecoder.from_pretrained(
                        str(Path(args.model_dir) / "rvq_depth_decoder"),
                        torch_dtype=torch.bfloat16,
                        local_files_only=True,
                    ).to(device)
                    depth_decoder.eval()
                frames, ended = _preroll_frames(
                    lm, depth_decoder, data["tokens"][0], args.endreg_frames, args.endreg_seed + index, device
                )
                blob = {"frame_embeds": None if frames is None else frames.cpu(), "ended": bool(ended)}
                torch.save(blob, cache_path)
            frame_embeds = None if blob["frame_embeds"] is None else blob["frame_embeds"].to(device)
            with torch.no_grad():
                prompt_embeds = lm.model.embed_tokens(data["tokens"][0])
                _last, base_margins, base_hidden = _forward_teacher_forced(lm, prompt_embeds, frame_embeds)
            data["prompt_embeds"] = prompt_embeds
            data["frame_embeds"] = frame_embeds
            data["base_margins"] = base_margins
            # Frame-position hiddens for --planreg_weight: the composition the
            # slider must not re-plan. Prompt positions excluded (the slider
            # target lives at prompt-last; causality keeps it frame-free).
            data["base_hidden"] = base_hidden[:, prompt_embeds.shape[1] :].float()
            ended_count += int(blob["ended"])
            n_frames = 0 if frame_embeds is None else frame_embeds.shape[1]
            print(
                f"row {index} endreg: frames={n_frames} "
                f"{'ended naturally' if blob['ended'] else 'hit the pre-roll cap'} "
                f"base end-margin first={base_margins[0].item():.2f} last={base_margins[-1].item():.2f}"
            )
        if depth_decoder is not None:
            del depth_decoder
            torch.cuda.empty_cache()

    network = LoRANetwork(
        lm,
        rank=args.rank,
        alpha=args.alpha,
        multiplier=1.0,
        delimiter="-",
        target_replace=TARGET_REPLACE,
        prefix="lora_te",
        train_method="full",
    ).to(device)
    n_mod = len(network.unet_loras)
    if n_mod == 0:
        raise RuntimeError("LoRA wrapped 0 Qwen3Attention linears")
    print(f"LM LoRA modules={n_mod} rank={args.rank}")
    for p in network.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=1e-6)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics = save_dir / f"{args.name}_train.jsonl"
    metrics_handle = metrics.open("w")

    history = []
    early_fired = False
    pbar = tqdm(range(args.steps), desc="lm-slider")
    for step in pbar:
        data = row_data[step % len(row_data)]
        neu_ids, neu_mask = data["tokens"]
        tgt_plus, tgt_minus, neu_ref = data["tgt_plus"], data["tgt_minus"], data["neu_ref"]

        if endreg:
            # One forward per pole over [prompt; composed frames]: causality
            # keeps the prompt-last hidden identical to a prompt-only forward,
            # so the slider loss is unchanged and the end margins come free.
            _set_scale(network, 1.0)
            pred_pos, m_pos, hid_pos = _forward_teacher_forced(lm, data["prompt_embeds"], data["frame_embeds"])
            ploss = F.mse_loss(pred_pos, tgt_plus)
            end_pos = F.mse_loss(m_pos, data["base_margins"])
            plan_pos = F.mse_loss(hid_pos[:, data["prompt_embeds"].shape[1] :].float(), data["base_hidden"])

            _set_scale(network, -1.0)
            pred_neg, m_neg, hid_neg = _forward_teacher_forced(lm, data["prompt_embeds"], data["frame_embeds"])
            nloss = F.mse_loss(pred_neg, tgt_minus)
            end_neg = F.mse_loss(m_neg, data["base_margins"])
            plan_neg = F.mse_loss(hid_neg[:, data["prompt_embeds"].shape[1] :].float(), data["base_hidden"])
            edrift_p = float((m_pos - data["base_margins"]).abs().mean().detach())
            edrift_n = float((m_neg - data["base_margins"]).abs().mean().detach())
            pdrift_p = float(plan_pos.detach())
            pdrift_n = float(plan_neg.detach())
        else:
            if args.planreg_weight > 0:
                raise ValueError("--planreg_weight requires the audio-end regularizer (its pre-roll supplies the plan)")
            _set_scale(network, 1.0)
            pred_pos = _encode_train(lm, neu_ids, neu_mask)
            ploss = F.mse_loss(pred_pos, tgt_plus)

            _set_scale(network, -1.0)
            pred_neg = _encode_train(lm, neu_ids, neu_mask)
            nloss = F.mse_loss(pred_neg, tgt_minus)
            end_pos = end_neg = torch.zeros((), device=device)
            plan_pos = plan_neg = torch.zeros((), device=device)
            edrift_p = edrift_n = 0.0
            pdrift_p = pdrift_n = 0.0

        v_pos = pred_pos - neu_ref
        v_neg = pred_neg - neu_ref
        v_pos_t = tgt_plus - neu_ref
        v_neg_t = tgt_minus - neu_ref
        cos_pos = F.cosine_similarity(v_pos, v_pos_t, dim=-1).mean()
        cos_neg = F.cosine_similarity(v_neg, v_neg_t, dim=-1).mean()
        collapse = F.cosine_similarity(v_pos, v_neg, dim=-1).mean()
        pperc = (torch.norm(pred_pos - tgt_plus) / torch.norm(v_pos_t).clamp_min(1e-6)).item()
        nperc = (torch.norm(pred_neg - tgt_minus) / torch.norm(v_neg_t).clamp_min(1e-6)).item()

        loss = ploss + nloss + 0.5 * args.endreg_weight * (end_pos + end_neg) + args.planreg_weight * (plan_pos + plan_neg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_value_(network.parameters(), clip_value=1.0)
        opt.step()

        row = {
            # 1-indexed, matching train_lora_music3.py: step N == N completed updates.
            "step": step + 1,
            "row": step % len(row_data),
            "loss": float(loss.detach()),
            "pperc": pperc,
            "nperc": nperc,
            "cos_pos": float(cos_pos.detach()),
            "cos_neg": float(cos_neg.detach()),
            "collapse": float(collapse.detach()),
            # Mean |end-margin drift| in nats along the teacher-forced roll.
            "edrift_p": edrift_p,
            "edrift_n": edrift_n,
            # Mean-squared hidden drift along the composed frames (planreg).
            "pdrift_p": pdrift_p,
            "pdrift_n": pdrift_n,
        }
        history.append(row)
        pbar.set_description(
            f"loss {row['loss']:.4f} p%{pperc*100:.1f} n%{nperc*100:.1f} "
            f"c+ {row['cos_pos']:.2f} c- {row['cos_neg']:.2f} col {row['collapse']:.2f} "
            f"e± {edrift_p:.2f}/{edrift_n:.2f}"
        )
        metrics_handle.write(json.dumps(row) + "\n")
        metrics_handle.flush()

        # The final step's weights already ship as _last.safetensors, so skip the
        # duplicate mid checkpoint there.
        if args.save_every and (step + 1) % args.save_every == 0 and (step + 1) < args.steps:
            network.save_weights(
                save_dir / f"{args.name}_step{step + 1}.safetensors", dtype=torch.float32
            )

        if (
            args.early_stop
            and len(history) >= args.min_steps
            and _early_stop_hit(history, args.early_window, args.early_cos, args.early_collapse, args.early_perc)
        ):
            early_fired = True
            print(
                f"early stop at step {step + 1} "
                f"(window={args.early_window} c+/{args.early_cos} col/{args.early_collapse} perc/{args.early_perc})"
            )
            break

    metrics_handle.close()
    last = save_dir / f"{args.name}_last.safetensors"
    network.save_weights(last, dtype=torch.float32)
    stopped = len(history)
    win = history[-args.early_window :] if len(history) >= args.early_window else history
    early_metrics = {
        key: (sum(r[key] for r in win) / len(win) if win else None)
        for key in ("cos_pos", "cos_neg", "collapse", "pperc", "nperc")
    }
    summary = {
        "schema": 3,
        "name": args.name,
        "checkpoint": str(last),
        "weights": str(last),
        "modules": n_mod,
        "rank": args.rank,
        "alpha": args.alpha,
        "steps": stopped,
        "steps_budget": args.steps,
        "lr": args.lr,
        "seed": int(args.seed),
        "rows": len(row_data),
        "symmetric": bool(args.symmetric),
        "common_beta": beta,
        "target_scale": float(args.target_scale),
        "planreg_weight": float(args.planreg_weight),
        "first": history[0] if history else None,
        "last": history[-1] if history else None,
        "target_replace": TARGET_REPLACE,
        "kind": "language_model",
        "prefix": "lora_te",
        "delimiter": "-",
        "train_method": "full",
        # Trained at ±1 by construction, so the user scale is the raw multiplier.
        "unit_scale": 1.0,
        "endreg": {
            "enabled": bool(endreg),
            "weight": float(args.endreg_weight),
            "frames": int(args.endreg_frames),
            "seed": int(args.endreg_seed),
            "rows_ended_naturally": ended_count if endreg else None,
            "final_edrift_p": (sum(r["edrift_p"] for r in win) / len(win)) if (endreg and win) else None,
            "final_edrift_n": (sum(r["edrift_n"] for r in win) / len(win)) if (endreg and win) else None,
        },
        "plus_label": prompts_meta.get("plus_label", ""),
        "minus_label": prompts_meta.get("minus_label", ""),
        "recommended_range": prompts_meta.get("recommended_range", [-2.0, 2.0]),
        "prompts_file": args.prompts_file,
        "early_stop": {
            "enabled": bool(args.early_stop),
            "fired": bool(early_fired),
            "window": args.early_window,
            "min_steps": args.min_steps,
            "cos": args.early_cos,
            "collapse": args.early_collapse,
            "perc": args.early_perc,
            "metrics": early_metrics,
        },
    }
    (save_dir / f"{args.name}_last.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return last


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="gender-lm")
    p.add_argument("--model_dir", default=str(DEFAULT_MODEL))
    p.add_argument("--prompts_file", required=True)
    p.add_argument("--save_dir", default="/ml2/music/sliders-conceptmod/models/gender-lm-slider")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=8.0)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--device", type=int, default=0)
    p.add_argument(
        "--seed",
        type=int,
        default=7,
        help="LoRA init seed; pin it so runs differing only in --seed are comparable",
    )
    p.add_argument(
        "--symmetric",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="antisymmetrize targets: tgt(±1) = neu ± (pos-neg)/2 (fixes +/- collapse)",
    )
    p.add_argument(
        "--common_beta",
        type=float,
        default=0.0,
        help="blend back beta*(midpoint - neutral) if symmetric ±1 sounds too weak",
    )
    p.add_argument(
        "--target_scale",
        type=float,
        default=1.0,
        help="train the symmetric unit against target_scale x the pole axis: a cooler-trained "
        "unit is a different delta shape, not a dial-down of the same one (relabeling the "
        "render scale is score-invariant; this is not that)",
    )
    p.add_argument(
        "--planreg_weight",
        type=float,
        default=0.0,
        help="penalize hidden-state drift along the teacher-forced composition (frame positions "
        "only, both poles): keeps the sampled arrangement near the base song while the slider "
        "still moves the prompt conditioning. Requires --endreg_weight > 0",
    )
    p.add_argument("--plus_label", default=None, help="override sidecar plus_label")
    p.add_argument("--minus_label", default=None, help="override sidecar minus_label")
    p.add_argument(
        "--endreg_weight",
        type=float,
        default=1.0,
        help="audio-end regularizer weight: keep log-odds of <|audio_end|> vs the semantic band "
        "unchanged along a base-model composition (0 disables; sliders trained without it "
        "measurably suppress endings and truncate at the duration cap)",
    )
    p.add_argument(
        "--endreg_frames",
        type=int,
        default=250,
        help="pre-roll length in audio frames (25/s) for the end regularizer; composed once per "
        "row with the base model and cached in --endreg_cache",
    )
    p.add_argument("--endreg_seed", type=int, default=7, help="pre-roll sampling seed (row index is added)")
    p.add_argument(
        "--endreg_cache",
        default=str(_REPO_ROOT / "cache" / "endreg"),
        help="directory for cached pre-rolls, keyed on model, prompt, frames and seed",
    )
    p.add_argument(
        "--early_stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stop when a rolling window matches c+/c-/collapse/perc (replayed on all v3 LM runs)",
    )
    p.add_argument("--early_window", type=int, default=50)
    p.add_argument("--min_steps", type=int, default=100, help="never stop before this many steps")
    p.add_argument("--early_cos", type=float, default=0.97, help="min mean c+ and c- in the window")
    p.add_argument("--early_collapse", type=float, default=-0.95, help="max mean collapse (more negative is better)")
    p.add_argument("--early_perc", type=float, default=0.20, help="max mean pperc/nperc in the window")
    args = p.parse_args()
    if args.steps < 1:
        p.error("--steps must be >= 1")
    return args


def _early_stop_hit(
    history: list[dict], window: int, min_cos: float, max_collapse: float, max_perc: float
) -> bool:
    if window <= 0 or len(history) < window:
        return False
    chunk = history[-window:]
    cos_pos = sum(r["cos_pos"] for r in chunk) / window
    cos_neg = sum(r["cos_neg"] for r in chunk) / window
    collapse = sum(r["collapse"] for r in chunk) / window
    pperc = sum(r["pperc"] for r in chunk) / window
    nperc = sum(r["nperc"] for r in chunk) / window
    return (
        cos_pos > min_cos
        and cos_neg > min_cos
        and collapse < max_collapse
        and pperc < max_perc
        and nperc < max_perc
    )


if __name__ == "__main__":
    train(parse_args())
