#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


METRICS = [
    "avg_bits_per_linear_param",
    "theoretical_linear_size_mb",
    "overall_tts_suitable_rate",
    "negative_subset_tts_suitable_rate",
    "overall_numbered_list_rate",
    "negative_subset_numbered_list_rate",
    "overall_emotion_match_rate",
    "negative_subset_emotion_match_rate",
    "overall_bad_format_rate",
    "negative_subset_bad_format_rate",
    "overall_avg_task_behavior_retention",
    "negative_subset_avg_task_behavior_retention",
    "overall_tts_retained_rate",
    "negative_subset_tts_retained_rate",
    "overall_emotion_match_retained_rate",
    "negative_subset_emotion_match_retained_rate",
    "overall_bad_format_increase_rate",
    "negative_subset_bad_format_increase_rate",
]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten(prefix, obj, out):
    for key, value in obj.items():
        if isinstance(value, (int, float, str)) or value is None:
            out[f"{prefix}_{key}"] = value


def parse_tag(tag):
    match = re.match(r"^random_k(\d+)_s(\d+)$", tag)
    if not match:
        return "", ""
    return match.group(1), match.group(2)


def read_seed_rows(root: Path):
    rows = []
    for report_path in sorted((root / "reports").glob("*_response_report.json")):
        tag = report_path.name[: -len("_response_report.json")]
        k, seed = parse_tag(tag)
        if not k:
            continue
        quant_path = root / "quant" / f"{tag}_quant_report.json"
        emotion_path = root / "emotion_match" / "reports" / f"{tag}_emotion_match_report.json"
        behavior_path = root / "behavior_alignment" / "reports" / f"{tag}_behavior_alignment_report.json"
        response = load_json(report_path)
        quant = load_json(quant_path) if quant_path.exists() else {}
        emotion = load_json(emotion_path) if emotion_path.exists() else {}
        behavior = load_json(behavior_path) if behavior_path.exists() else {}
        row = {
            "row_type": "seed",
            "tag": tag,
            "strategy": "random_layercount",
            "layer_count": int(k),
            "seed": int(seed),
            "avg_bits_per_linear_param": quant.get("avg_bits_per_linear_param", ""),
            "theoretical_linear_size_mb": quant.get("theoretical_linear_size_mb", ""),
            "protected_layers": " ".join(str(x) for x in quant.get("protected_layers", [])),
        }
        flatten("overall", response.get("overall", {}), row)
        flatten("negative_subset", response.get("negative_subset", {}), row)
        flatten("overall", emotion.get("overall", {}), row)
        flatten("negative_subset", emotion.get("negative_subset", {}), row)
        flatten("overall", behavior.get("overall", {}), row)
        flatten("negative_subset", behavior.get("negative_subset", {}), row)
        rows.append(row)
    return rows


def as_float(value):
    try:
        if value == "" or value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def mean_std(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return "", ""
    avg = sum(vals) / len(vals)
    if len(vals) == 1:
        return avg, 0.0
    var = sum((v - avg) ** 2 for v in vals) / (len(vals) - 1)
    return avg, math.sqrt(var)


def aggregate_rows(seed_rows):
    grouped = defaultdict(list)
    for row in seed_rows:
        grouped[row["layer_count"]].append(row)
    rows = []
    for layer_count, members in sorted(grouped.items()):
        out = {
            "row_type": "mean_std",
            "tag": f"random_k{layer_count}_mean_std",
            "strategy": "random_layercount",
            "layer_count": layer_count,
            "seed": "mean_std",
            "n_seeds": len(members),
            "protected_layers": "",
        }
        for metric in METRICS:
            avg, std = mean_std([as_float(row.get(metric)) for row in members])
            out[f"{metric}_mean"] = avg
            out[f"{metric}_std"] = std
            out[metric] = avg
        rows.append(out)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Collect random exact-layer-count held-out test results.")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--csv", type=Path, required=True)
    args = ap.parse_args()

    seed_rows = read_seed_rows(args.root)
    rows = seed_rows + aggregate_rows(seed_rows)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "row_type",
        "tag",
        "strategy",
        "layer_count",
        "seed",
        "n_seeds",
        "avg_bits_per_linear_param",
        "theoretical_linear_size_mb",
        "overall_tts_suitable_rate",
        "negative_subset_tts_suitable_rate",
        "overall_emotion_match_rate",
        "negative_subset_emotion_match_rate",
        "overall_bad_format_rate",
        "negative_subset_bad_format_rate",
        "negative_subset_avg_task_behavior_retention",
        "negative_subset_tts_retained_rate",
        "negative_subset_emotion_match_retained_rate",
        "negative_subset_bad_format_increase_rate",
        "protected_layers",
    ]
    ordered = [x for x in preferred if x in fieldnames] + [x for x in fieldnames if x not in preferred]
    with open(args.csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] {args.csv}")
    print(f"[ROWS] {len(rows)}")


if __name__ == "__main__":
    main()
