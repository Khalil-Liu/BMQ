#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import CLASS_WEIGHTS, iter_jsonl, normalize_label


SYSTEM_PROMPT = (
    "你是一个情绪识别器。请只输出一个情绪标签，必须是："
    "happy, neutral, sad, anxious, angry, lonely, tired_low, comfort_need。"
)


def pick_dtype(args):
    if args.force_float32:
        return torch.float32
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return "auto"


def load_model(args):
    if args.local_files_only:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kwargs = {
        "pretrained_model_name_or_path": args.model,
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }
    if args.device == "auto":
        load_kwargs["device_map"] = "auto"
    try:
        model = AutoModelForCausalLM.from_pretrained(**load_kwargs, dtype=pick_dtype(args))
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(**load_kwargs, torch_dtype=pick_dtype(args))
    if args.device in {"cuda", "cpu"}:
        model = model.to(args.device)
    model.train()
    model.config.use_cache = False
    return model, tokenizer


def build_text(tokenizer, user_text: str, label: str) -> Tuple[str, str]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    answer = label
    return prompt, prompt + answer


def compute_loss(model, tokenizer, user_text: str, label: str):
    prompt, full = build_text(tokenizer, user_text, label)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    batch = tokenizer(full, add_special_tokens=False, return_tensors="pt")
    batch = {k: v.to(model.device) for k, v in batch.items()}
    labels = batch["input_ids"].clone()
    labels[:, : min(len(prompt_ids), labels.shape[1])] = -100
    out = model(**batch, labels=labels)
    return out.loss


def layer_key(name: str) -> str:
    match = re.search(r"(model\.)?layers\.(\d+)", name)
    if match:
        return f"layers.{match.group(2)}"
    if "embed_tokens" in name:
        return "embed_tokens"
    if "lm_head" in name:
        return "lm_head"
    return "other"


def module_key(name: str) -> str:
    if any(k in name for k in ("q_proj", "k_proj", "v_proj", "o_proj", "self_attn")):
        return "attention"
    if any(k in name for k in ("gate_proj", "up_proj", "down_proj", "mlp")):
        return "mlp"
    if "embed_tokens" in name:
        return "embedding"
    if "lm_head" in name:
        return "lm_head"
    return "other"


def accumulate_scores(model, layer_scores: Dict[str, float], module_scores: Dict[str, float]):
    for name, param in model.named_parameters():
        if param.grad is None or not param.requires_grad:
            continue
        score = (param.grad.detach().float() * param.detach().float()).abs().sum().item()
        layer_scores[layer_key(name)] += score
        module_scores[module_key(name)] += score


def normalize_scores(scores: Dict[str, float]):
    total = sum(scores.values()) or 1.0
    rows = []
    for rank, (name, score) in enumerate(sorted(scores.items(), key=lambda x: x[1], reverse=True), start=1):
        rows.append(
            {
                "name": name,
                "score": score,
                "normalized_score": score / total,
                "rank": rank,
            }
        )
    return rows


def write_csv(path: Path, rows, name_field: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[name_field, "score", "normalized_score", "rank"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    name_field: row["name"],
                    "score": row["score"],
                    "normalized_score": row["normalized_score"],
                    "rank": row["rank"],
                }
            )


def parse_args():
    ap = argparse.ArgumentParser(description="Compute EAAS gradient sensitivity scores for emotion-label loss.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--max_samples", type=int, default=32)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--force_float32", action="store_true")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--local_files_only", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    model, tokenizer = load_model(args)
    rows = list(iter_jsonl(args.input))[: args.max_samples]
    layer_scores = defaultdict(float)
    module_scores = defaultdict(float)
    loss_sum = 0.0

    for idx, row in enumerate(rows, start=1):
        label = normalize_label(row.get("label") or row.get("emotion_label"))
        text = row.get("text", "")
        weight = CLASS_WEIGHTS.get(label, 1.0)
        model.zero_grad(set_to_none=True)
        loss = compute_loss(model, tokenizer, text, label) * weight
        loss.backward()
        accumulate_scores(model, layer_scores, module_scores)
        loss_sum += float(loss.detach().float().item())
        print(f"[{idx}/{len(rows)}] label={label} weight={weight} loss={loss_sum/idx:.4f}")

    layer_rows = normalize_scores(layer_scores)
    module_rows = normalize_scores(module_scores)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "eaas_layer_scores.json").write_text(
        json.dumps(layer_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.out_dir / "eaas_module_scores.json").write_text(
        json.dumps(module_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(args.out_dir / "eaas_layer_scores.csv", layer_rows, "layer_name")
    write_csv(args.out_dir / "eaas_module_scores.csv", module_rows, "module_type")
    summary = {
        "method": "EAAS",
        "samples": len(rows),
        "avg_weighted_loss": loss_sum / max(len(rows), 1),
        "top_layers": layer_rows[:10],
        "module_scores": module_rows,
    }
    (args.out_dir / "eaas_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] {args.out_dir}")


if __name__ == "__main__":
    main()
