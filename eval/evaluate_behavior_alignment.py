#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from statistics import mean

from common import LABEL_ORDER, NEGATIVE_LABELS, iter_jsonl, normalize_label
from eval.evaluate_emotion_match import score_row as emotion_match_metrics
from eval.evaluate_response_deployment import response_metrics


def row_key(row, fallback_idx):
    return str(row.get("id", fallback_idx))


def load_rows(path: Path):
    rows = []
    for idx, row in enumerate(iter_jsonl(path), start=1):
        rows.append((row_key(row, idx), row))
    return rows


def align_rows(ref_path: Path, pred_path: Path):
    ref_rows = load_rows(ref_path)
    pred_rows = load_rows(pred_path)
    ref_by_key = {key: row for key, row in ref_rows}
    aligned = []
    allow_index_fallback = len(ref_rows) == len(pred_rows)
    for idx, (key, pred) in enumerate(pred_rows, start=1):
        ref = ref_by_key.get(key)
        if ref is None and allow_index_fallback and idx <= len(ref_rows):
            ref = ref_rows[idx - 1][1]
        if ref is not None:
            aligned.append((ref, pred))
    return aligned


def bool_retention(ref_value, pred_value):
    if ref_value:
        return bool(pred_value)
    return None


def score_pair(ref, pred):
    label = normalize_label(pred.get("label") or ref.get("label"))
    ref_resp = response_metrics(ref)
    pred_resp = response_metrics(pred)
    ref_match = emotion_match_metrics(ref)
    pred_match = emotion_match_metrics(pred)

    ref_tts = bool(ref_resp["tts_suitable"])
    pred_tts = bool(pred_resp["tts_suitable"])
    ref_emotion = bool(ref_match["emotion_match"])
    pred_emotion = bool(pred_match["emotion_match"])
    ref_bad = bool(ref_match["bad_format"])
    pred_bad = bool(pred_match["bad_format"])
    ref_numbered = bool(ref_resp["numbered_list"])
    pred_numbered = bool(pred_resp["numbered_list"])
    ref_abnormal = bool(ref_resp["abnormal"])
    pred_abnormal = bool(pred_resp["abnormal"])

    tts_retained = bool_retention(ref_tts, pred_tts)
    emotion_retained = bool_retention(ref_emotion, pred_emotion)
    good_format_retained = bool_retention(not ref_bad, not pred_bad)
    normal_retained = bool_retention(not ref_abnormal, not pred_abnormal)

    task_flags = [x for x in [tts_retained, emotion_retained, good_format_retained, normal_retained] if x is not None]

    return {
        "label": label,
        "n": 1,
        "tts_agreement": ref_tts == pred_tts,
        "tts_retained": tts_retained,
        "emotion_match_agreement": ref_emotion == pred_emotion,
        "emotion_match_retained": emotion_retained,
        "good_format_agreement": (not ref_bad) == (not pred_bad),
        "good_format_retained": good_format_retained,
        "normal_response_retained": normal_retained,
        "numbered_list_changed": ref_numbered != pred_numbered,
        "numbered_list_increase": (not ref_numbered) and pred_numbered,
        "bad_format_increase": (not ref_bad) and pred_bad,
        "abnormal_increase": (not ref_abnormal) and pred_abnormal,
        "task_behavior_retention": sum(bool(x) for x in task_flags) / len(task_flags) if task_flags else 0.0,
        "ref_tts_suitable": ref_tts,
        "pred_tts_suitable": pred_tts,
        "ref_emotion_match": ref_emotion,
        "pred_emotion_match": pred_emotion,
        "ref_bad_format": ref_bad,
        "pred_bad_format": pred_bad,
        "ref_response_chars": len(str(ref.get("response") or "")),
        "pred_response_chars": len(str(pred.get("response") or "")),
    }


def aggregate(rows):
    if not rows:
        return {}
    out = {"n": len(rows)}
    keys = [k for k in rows[0].keys() if k not in {"label", "n"}]
    for key in keys:
        vals = [row[key] for row in rows if row.get(key) is not None]
        if not vals:
            continue
        if isinstance(vals[0], bool):
            out[f"{key}_rate"] = sum(bool(v) for v in vals) / len(vals)
        elif isinstance(vals[0], (int, float)):
            out[f"avg_{key}"] = mean(vals)
    return out


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate task behavior alignment between FP16 reference and quantized responses.")
    ap.add_argument("--ref", type=Path, required=True, help="FP16/reference response jsonl")
    ap.add_argument("--pred", type=Path, required=True, help="Quantized response jsonl")
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--detail_jsonl", type=Path, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    aligned = align_rows(args.ref, args.pred)
    metric_rows = []
    negative_rows = []
    by_label = defaultdict(list)
    for ref, pred in aligned:
        metrics = score_pair(ref, pred)
        label = metrics["label"]
        metric_rows.append(metrics)
        by_label[label].append(metrics)
        if label in NEGATIVE_LABELS:
            negative_rows.append(metrics)

    report = {
        "ref": str(args.ref),
        "pred": str(args.pred),
        "overall": aggregate(metric_rows),
        "negative_subset": aggregate(negative_rows),
        "by_label": {label: aggregate(by_label[label]) for label in LABEL_ORDER},
        "definition": {
            "task_behavior_retention": "Mean retained rate over reference-positive TTS, emotion match, good format, and normal response flags.",
            "retained": "A behavior is retained when it is true in the FP16 reference and remains true in the quantized output.",
            "increase": "A failure increases when it is false in the FP16 reference and true in the quantized output.",
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["group"] + sorted(report["overall"].keys())
    with open(args.csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"group": "overall", **report["overall"]})
        writer.writerow({"group": "negative_subset", **report["negative_subset"]})
        for label in LABEL_ORDER:
            writer.writerow({"group": label, **report["by_label"][label]})

    if args.detail_jsonl:
        args.detail_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(args.detail_jsonl, "w", encoding="utf-8") as f:
            for row in metric_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[REPORT] {args.report}")
    print(f"[CSV] {args.csv}")
    print(
        "[SUMMARY] "
        f"ret={report['overall'].get('avg_task_behavior_retention', 0):.4f} "
        f"neg_ret={report['negative_subset'].get('avg_task_behavior_retention', 0):.4f} "
        f"tts_ret={report['overall'].get('tts_retained_rate', 0):.4f} "
        f"emo_ret={report['overall'].get('emotion_match_retained_rate', 0):.4f}"
    )


if __name__ == "__main__":
    main()


