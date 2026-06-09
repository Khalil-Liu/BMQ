#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch

from common import LABEL_ORDER, NEGATIVE_LABELS, iter_jsonl, normalize_label
from eval.evaluate_emotion_match import score_row as emotion_match_metrics
from eval.evaluate_response_deployment import response_metrics
from infer import (
    SYSTEM_PROMPT,
    fake_quant_tensor,
    generate,
    layer_index,
    load_inputs,
    load_model,
    module_type,
)


def parse_layer_list(value):
    if not value or value == "all":
        return None
    return {int(x.strip()) for x in value.split(",") if x.strip()}


def normalize_ref_row(obj, idx):
    return {
        "id": obj.get("id", idx),
        "text": obj["text"],
        "label": normalize_label(obj.get("label")),
        "response": obj["response"],
    }


def load_ref_rows(path: Path):
    rows = []
    for idx, obj in enumerate(iter_jsonl(path), start=1):
        if obj.get("text") and obj.get("response"):
            rows.append(normalize_ref_row(obj, idx))
    return rows


def align_ref_rows(path: Path, target_rows):
    ref_rows = load_ref_rows(path)
    by_id = {str(row.get("id")): row for row in ref_rows}
    by_text = {str(row.get("text")): row for row in ref_rows}
    aligned = []
    missing = 0
    for idx, row in enumerate(target_rows):
        ref = by_id.get(str(row.get("id"))) or by_text.get(str(row.get("text")))
        if ref is None and idx < len(ref_rows):
            ref = ref_rows[idx]
        if ref is None:
            missing += 1
            continue
        aligned.append(ref)
    if missing:
        print(f"[WARN] Missing {missing} reference rows during alignment.")
    return aligned


def score_rows(rows):
    metric_rows = []
    negative_rows = []
    by_label = defaultdict(list)
    for row in rows:
        label = normalize_label(row.get("label"))
        resp = response_metrics(row)
        match = emotion_match_metrics(row)
        metrics = {
            "tts": bool(resp["tts_suitable"]),
            "numbered": bool(resp["numbered_list"]),
            "bad_format": bool(match["bad_format"]),
            "emotion_match": bool(match["emotion_match"]),
            "abnormal": bool(resp["abnormal"]),
        }
        metric_rows.append(metrics)
        by_label[label].append(metrics)
        if label in NEGATIVE_LABELS:
            negative_rows.append(metrics)
    return {
        "overall": aggregate(metric_rows),
        "negative_subset": aggregate(negative_rows),
        "by_label": {label: aggregate(by_label[label]) for label in LABEL_ORDER},
    }


def aggregate(rows):
    if not rows:
        return {}
    out = {"n": len(rows)}
    for key in rows[0].keys():
        vals = [row[key] for row in rows]
        if isinstance(vals[0], bool):
            out[f"{key}_rate"] = sum(bool(v) for v in vals) / len(vals)
        elif isinstance(vals[0], (int, float)):
            out[f"avg_{key}"] = mean(vals)
    return out


def discover_layers(model, module_kind):
    layers = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and layer_index(name) is not None:
            if module_kind == "all" or module_type(name) == module_kind:
                layers.add(layer_index(name))
    return sorted(layers)


def target_modules(model, layer, module_kind):
    modules = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if layer_index(name) != layer:
            continue
        if module_kind != "all" and module_type(name) != module_kind:
            continue
        modules.append((name, module))
    return modules


def quantize_temporarily(modules, bits, group_size):
    originals = []
    with torch.no_grad():
        for name, module in modules:
            originals.append((module, module.weight.data.detach().clone()))
            module.weight.data.copy_(fake_quant_tensor(module.weight.data, bits, group_size))
    return originals


def restore_modules(originals):
    with torch.no_grad():
        for module, weight in originals:
            module.weight.data.copy_(weight)


def behavior_drop(ref_report, quant_report, weights):
    ref_neg = ref_report["negative_subset"]
    q_neg = quant_report["negative_subset"]
    ref_over = ref_report["overall"]
    q_over = quant_report["overall"]

    neg_tts_drop = max(0.0, ref_neg.get("tts_rate", 0.0) - q_neg.get("tts_rate", 0.0))
    neg_match_drop = max(0.0, ref_neg.get("emotion_match_rate", 0.0) - q_neg.get("emotion_match_rate", 0.0))
    neg_bad_inc = max(0.0, q_neg.get("bad_format_rate", 0.0) - ref_neg.get("bad_format_rate", 0.0))
    over_bad_inc = max(0.0, q_over.get("bad_format_rate", 0.0) - ref_over.get("bad_format_rate", 0.0))
    score = (
        weights["neg_tts"] * neg_tts_drop
        + weights["neg_match"] * neg_match_drop
        + weights["neg_bad"] * neg_bad_inc
        + weights["overall_bad"] * over_bad_inc
    )
    return {
        "behavior_drop_score": score,
        "neg_tts_drop": neg_tts_drop,
        "neg_match_drop": neg_match_drop,
        "neg_bad_format_increase": neg_bad_inc,
        "overall_bad_format_increase": over_bad_inc,
        "quant_neg_tts": q_neg.get("tts_rate", 0.0),
        "quant_neg_match": q_neg.get("emotion_match_rate", 0.0),
        "quant_neg_bad_format": q_neg.get("bad_format_rate", 0.0),
        "quant_overall_bad_format": q_over.get("bad_format_rate", 0.0),
    }


def parse_args():
    ap = argparse.ArgumentParser(description="Measure layer-wise behavior drop by temporarily quantizing one layer/module group.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True, help="FP16 reference response jsonl")
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument("--out_json", type=Path, required=True)
    ap.add_argument("--module_type", choices=["mlp", "attention", "all"], default="mlp")
    ap.add_argument("--layers", default="all", help="Comma-separated layer ids, or all")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--system_prompt", default=SYSTEM_PROMPT)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--force_float32", action="store_true")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--neg_tts_weight", type=float, default=1.0)
    ap.add_argument("--neg_match_weight", type=float, default=1.0)
    ap.add_argument("--neg_bad_weight", type=float, default=0.5)
    ap.add_argument("--overall_bad_weight", type=float, default=0.5)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.local_files_only:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model, tokenizer = load_model(args)
    rows = load_inputs(args.input)
    if args.limit > 0:
        rows = rows[: args.limit]
    ref_rows = align_ref_rows(args.ref, rows)
    ref_report = score_rows(ref_rows)

    requested = parse_layer_list(args.layers)
    layers = discover_layers(model, args.module_type)
    if requested is not None:
        layers = [x for x in layers if x in requested]

    weights = {
        "neg_tts": args.neg_tts_weight,
        "neg_match": args.neg_match_weight,
        "neg_bad": args.neg_bad_weight,
        "overall_bad": args.overall_bad_weight,
    }
    results = []
    for layer in layers:
        modules = target_modules(model, layer, args.module_type)
        param_count = sum(module.weight.numel() for _, module in modules)
        originals = quantize_temporarily(modules, args.bits, args.group_size)
        pred_rows = []
        try:
            for i, row in enumerate(rows, start=1):
                pred = generate(model, tokenizer, row["text"], args)
                pred_rows.append({**row, **pred})
                print(f"[layer {layer}] [{i}/{len(rows)}] label={row['label']}")
        finally:
            restore_modules(originals)
        quant_report = score_rows(pred_rows)
        drop = behavior_drop(ref_report, quant_report, weights)
        result = {
            "layer": layer,
            "module_type": args.module_type,
            "bits": args.bits,
            "params": param_count,
            **drop,
        }
        results.append(result)
        print(
            f"[LAYER] {layer} score={drop['behavior_drop_score']:.4f} "
            f"neg_tts_drop={drop['neg_tts_drop']:.4f} neg_match_drop={drop['neg_match_drop']:.4f}"
        )

    max_score = max((r["behavior_drop_score"] for r in results), default=0.0)
    for rank, row in enumerate(sorted(results, key=lambda x: x["behavior_drop_score"], reverse=True), start=1):
        row["rank"] = rank
        row["normalized_behavior_drop"] = row["behavior_drop_score"] / max_score if max_score > 0 else 0.0

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "layer",
        "module_type",
        "bits",
        "params",
        "behavior_drop_score",
        "normalized_behavior_drop",
        "neg_tts_drop",
        "neg_match_drop",
        "neg_bad_format_increase",
        "overall_bad_format_increase",
        "quant_neg_tts",
        "quant_neg_match",
        "quant_neg_bad_format",
        "quant_overall_bad_format",
    ]
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda x: x["rank"]))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps({"reference": ref_report, "weights": weights, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[CSV] {args.out_csv}")
    print(f"[JSON] {args.out_json}")


if __name__ == "__main__":
    main()

