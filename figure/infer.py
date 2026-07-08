#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import os
import random
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import iter_jsonl, normalize_label


SYSTEM_PROMPT = (
    "你是一个中文情感陪伴机器人。用户会直接说出自己的困扰、压力或日常感受。"
    "请用自然、真诚、温和的中文回应：先接住对方的感受，再给一个很小、可执行的建议，"
    "必要时问一个轻量的问题继续陪伴。不要说教，不要输出动作标签。"
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
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=args.local_files_only)
    except AttributeError as exc:
        if "extra_special_tokens" not in str(exc) and "keys" not in str(exc):
            raise
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
            extra_special_tokens={},
        )
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
    model.eval()
    if not args.do_sample:
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
    return model, tokenizer


def layer_index(name: str):
    match = re.search(r"layers\.(\d+)\.", name)
    return None if match is None else int(match.group(1))


def module_type(name: str):
    if ".mlp." in name:
        return "mlp"
    if ".self_attn." in name or ".attention." in name:
        return "attention"
    if "lm_head" in name:
        return "lm_head"
    return "other"


def read_eaas_top_layers(path: Path, top_ratio: float):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("layer_name", "")
            if name.startswith("layers."):
                rows.append((int(name.split(".")[1]), int(row.get("rank", 999999))))
    rows.sort(key=lambda x: x[1])
    k = max(1, math.ceil(len(rows) * top_ratio))
    return {idx for idx, _ in rows[:k]}


def read_protected_layers(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return set()
    layers = set()
    if path.suffix.lower() == ".json":
        obj = json.loads(text)
        values = obj.get("protected_layers", obj if isinstance(obj, list) else [])
        return {int(x) for x in values}
    if path.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                value = row.get("layer") or row.get("layer_idx") or row.get("layer_name")
                if value is None:
                    continue
                value = str(value)
                if value.startswith("layers."):
                    value = value.split(".")[1]
                layers.add(int(value))
        return layers
    for part in re.split(r"[\s,]+", text):
        if part:
            layers.add(int(part))
    return layers


def random_top_layers(model, top_ratio: float, seed: int):
    layers = sorted({layer_index(name) for name, module in model.named_modules() if isinstance(module, torch.nn.Linear) and layer_index(name) is not None})
    rng = random.Random(seed)
    k = max(1, math.ceil(len(layers) * top_ratio))
    return set(rng.sample(layers, k))


def fake_quant_tensor(tensor: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    if bits >= 16:
        return tensor
    original_dtype = tensor.dtype
    x = tensor.detach().float()
    if x.ndim < 2:
        max_abs = x.abs().max().clamp_min(1e-8)
        qmax = (2 ** (bits - 1)) - 1
        scale = max_abs / qmax
        return (torch.clamp(torch.round(x / scale), -qmax, qmax) * scale).to(original_dtype)

    out_features, in_features = x.shape[0], x.shape[1]
    flat = x.reshape(out_features, in_features, -1)
    deq = torch.empty_like(flat)
    qmax = (2 ** (bits - 1)) - 1
    for start in range(0, in_features, group_size):
        end = min(start + group_size, in_features)
        block = flat[:, start:end, :]
        max_abs = block.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
        scale = max_abs / qmax
        deq[:, start:end, :] = torch.clamp(torch.round(block / scale), -qmax, qmax) * scale
    return deq.reshape_as(x).to(original_dtype)


def apply_fake_quant(model, args):
    if args.strategy in {"uniform_int4", "uniform_int8"}:
        protected_layers = set()
    elif args.strategy == "random_mlp_int8":
        protected_layers = random_top_layers(model, args.top_ratio, args.seed)
    elif args.strategy == "eaas_mlp_int8":
        if not args.eaas_layer_scores:
            raise ValueError("--eaas_layer_scores is required for eaas_mlp_int8")
        protected_layers = read_eaas_top_layers(args.eaas_layer_scores, args.top_ratio)
    elif args.strategy == "custom_mlp_int8":
        if not args.protected_layers:
            raise ValueError("--protected_layers is required for custom_mlp_int8")
        protected_layers = read_protected_layers(args.protected_layers)
    else:
        raise ValueError(f"unknown strategy: {args.strategy}")

    stats = {"int4_params": 0, "int8_params": 0, "fp_params": 0, "protected_layers": sorted(protected_layers), "modules": []}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        idx = layer_index(name)
        mtype = module_type(name)
        use_int8 = args.strategy == "uniform_int8" or (
            args.strategy != "uniform_int4" and mtype == "mlp" and idx in protected_layers
        )
        bits = 8 if use_int8 else 4
        with torch.no_grad():
            module.weight.data.copy_(fake_quant_tensor(module.weight.data, bits, args.group_size))
            if module.bias is not None and args.quantize_bias:
                module.bias.data.copy_(fake_quant_tensor(module.bias.data, bits, args.group_size))
        params = module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
        if bits == 8:
            stats["int8_params"] += params
        elif bits == 4:
            stats["int4_params"] += params
        else:
            stats["fp_params"] += params
        stats["modules"].append({"name": name, "layer": idx, "type": mtype, "bits": bits, "params": params})
    total_bits = stats["int4_params"] * 4 + stats["int8_params"] * 8 + stats["fp_params"] * 16
    total_params = stats["int4_params"] + stats["int8_params"] + stats["fp_params"]
    stats["linear_params"] = total_params
    stats["avg_bits_per_linear_param"] = total_bits / max(total_params, 1)
    stats["theoretical_linear_size_mb"] = total_bits / 8 / 1024 / 1024
    return stats


def build_prompt(tokenizer, text: str, system_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate(model, tokenizer, text: str, args):
    prompt = build_prompt(tokenizer, text, args.system_prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "repetition_penalty": args.repetition_penalty,
    }
    if args.do_sample:
        gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)
    else:
        gen_kwargs.update(do_sample=False)
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    total_ms = (time.perf_counter() - start) * 1000.0
    gen_ids = out[0][inputs["input_ids"].shape[1] :]
    output_tokens = int(gen_ids.shape[0])
    response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return {
        "response": response,
        "input_tokens": int(inputs["input_ids"].shape[1]),
        "output_tokens": output_tokens,
        "total_ms": total_ms,
        "ms_per_output_token": total_ms / max(output_tokens, 1),
        "tokens_per_second": 1000.0 * output_tokens / max(total_ms, 1e-9),
    }


def load_inputs(path: Path):
    rows = []
    for idx, obj in enumerate(iter_jsonl(path), start=1):
        text = obj.get("text")
        label = obj.get("label") or obj.get("emotion_label")
        if text:
            row = {"id": obj.get("id", idx), "text": text, "label": normalize_label(label)}
            for key in ("wav_path", "source", "emotion_category", "speaker"):
                if key in obj:
                    row[key] = obj[key]
            rows.append(row)
    return rows


def parse_args():
    ap = argparse.ArgumentParser(description="Generate responses after PyTorch-side fake mixed precision quantization.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--quant_report", type=Path, required=True)
    ap.add_argument("--strategy", choices=["uniform_int4", "uniform_int8", "random_mlp_int8", "eaas_mlp_int8", "custom_mlp_int8"], required=True)
    ap.add_argument("--eaas_layer_scores", type=Path, default=None)
    ap.add_argument("--protected_layers", type=Path, default=None)
    ap.add_argument("--top_ratio", type=float, default=0.3)
    ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--system_prompt", default=SYSTEM_PROMPT)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--force_float32", action="store_true")
    ap.add_argument("--quantize_bias", action="store_true")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    model, tokenizer = load_model(args)
    quant_stats = apply_fake_quant(model, args)
    args.quant_report.parent.mkdir(parents=True, exist_ok=True)
    args.quant_report.write_text(json.dumps(quant_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[QUANT] strategy={args.strategy} avg_bits={quant_stats['avg_bits_per_linear_param']:.3f} "
        f"linear_size_mb={quant_stats['theoretical_linear_size_mb']:.2f} protected_layers={quant_stats['protected_layers']}"
    )

    rows = load_inputs(args.input)
    if args.limit > 0:
        rows = rows[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows, start=1):
            pred = generate(model, tokenizer, row["text"], args)
            out = {**row, **pred, "fake_quant_strategy": args.strategy}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            print(
                f"[{i}/{len(rows)}] label={row['label']} out_tok={pred['output_tokens']} "
                f"ms/tok={pred['ms_per_output_token']:.2f}"
            )
    print(f"[DONE] {args.output}")


if __name__ == "__main__":
    main()
