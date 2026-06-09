#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
from pathlib import Path


def minmax(values):
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if hi <= lo:
        return {k: 0.0 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def read_eaas(path: Path):
    out = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("layer_name", "")
            if name.startswith("layers."):
                layer = int(name.split(".")[1])
                out[layer] = float(row.get("normalized_score") or row.get("score") or 0.0)
    return out


def read_behavior(path: Path):
    out = {}
    params = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer = int(row["layer"])
            out[layer] = float(row.get("normalized_behavior_drop") or row.get("behavior_drop_score") or 0.0)
            params[layer] = int(float(row.get("params") or 0))
    return out, params


def parse_budgets(value):
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def write_layers(path: Path, layers):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(",".join(str(x) for x in sorted(layers)), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Build EBS = alpha*EAAS + beta*BehaviorDrop and export budget-aware protected layer sets.")
    ap.add_argument("--eaas", type=Path, required=True)
    ap.add_argument("--behavior", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--budgets", default="4.64,4.96", help="Average bit budgets for Linear params")
    ap.add_argument("--base_bits", type=float, default=4.0)
    ap.add_argument("--protect_bits", type=float, default=8.0)
    args = ap.parse_args()

    eaas = minmax(read_eaas(args.eaas))
    behavior, params = read_behavior(args.behavior)
    behavior = minmax(behavior)
    layers = sorted(set(eaas) | set(behavior))

    rows = []
    for layer in layers:
        e = eaas.get(layer, 0.0)
        b = behavior.get(layer, 0.0)
        p = params.get(layer, 0)
        score = args.alpha * e + args.beta * b
        extra_cost = p * (args.protect_bits - args.base_bits)
        rows.append(
            {
                "layer": layer,
                "eaas_norm": e,
                "behavior_norm": b,
                "ebs_score": score,
                "params": p,
                "extra_bit_cost": extra_cost,
                "score_per_cost": score / extra_cost if extra_cost > 0 else score,
            }
        )
    rows.sort(key=lambda x: x["ebs_score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["ebs_rank"] = rank

    args.out_dir.mkdir(parents=True, exist_ok=True)
    score_csv = args.out_dir / "ebs_layer_scores.csv"
    with open(score_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["ebs_rank", "layer", "ebs_score", "eaas_norm", "behavior_norm", "params", "extra_bit_cost", "score_per_cost"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_params = sum(params.values()) or len(layers)
    allocs = []
    for budget in parse_budgets(args.budgets):
        ratio = (budget - args.base_bits) / max(args.protect_bits - args.base_bits, 1e-9)
        target_params = max(0, ratio * total_params)
        selected = []
        used = 0
        for row in rows:
            if used >= target_params:
                break
            selected.append(row["layer"])
            used += row.get("params", 0) or total_params / max(len(layers), 1)
        tag = f"avg{str(budget).replace('.', '')}"
        layer_path = args.out_dir / f"ebs_protected_layers_{tag}.txt"
        write_layers(layer_path, selected)
        actual_avg_bits = args.base_bits + (args.protect_bits - args.base_bits) * (used / total_params if total_params else 0)
        allocs.append(
            {
                "budget_avg_bits": budget,
                "actual_avg_bits": actual_avg_bits,
                "target_protected_param_ratio": ratio,
                "actual_protected_param_ratio": used / total_params if total_params else 0,
                "protected_layers": " ".join(str(x) for x in sorted(selected)),
                "protected_layers_file": str(layer_path),
            }
        )

    alloc_csv = args.out_dir / "ebs_allocations.csv"
    with open(alloc_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "budget_avg_bits",
            "actual_avg_bits",
            "target_protected_param_ratio",
            "actual_protected_param_ratio",
            "protected_layers",
            "protected_layers_file",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(allocs)

    summary = {"alpha": args.alpha, "beta": args.beta, "scores": str(score_csv), "allocations": str(alloc_csv)}
    (args.out_dir / "ebs_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SCORES] {score_csv}")
    print(f"[ALLOC] {alloc_csv}")


if __name__ == "__main__":
    main()
