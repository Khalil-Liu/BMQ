#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


LABEL_ORDER = [
    "happy",
    "neutral",
    "sad",
    "anxious",
    "angry",
    "lonely",
    "tired_low",
    "comfort_need",
]

NEGATIVE_LABELS = {"sad", "anxious", "lonely", "comfort_need"}

LABEL_ALIASES = {
    "happy": "happy",
    "joy": "happy",
    "love": "happy",
    "开心/愉悦": "happy",
    "neutral": "neutral",
    "中性/日常": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "悲伤/难过": "sad",
    "anxious": "anxious",
    "fear": "anxious",
    "overwhelmed": "anxious",
    "焦虑/压力": "anxious",
    "angry": "angry",
    "anger": "angry",
    "生气/烦躁": "angry",
    "lonely": "lonely",
    "孤独/需要陪伴": "lonely",
    "tired": "tired_low",
    "tired_low": "tired_low",
    "burnout": "tired_low",
    "低落/疲惫": "tired_low",
    "comfort_need": "comfort_need",
    "calm": "comfort_need",
    "安慰/放松需求": "comfort_need",
}

CLASS_WEIGHTS = {
    "happy": 1.0,
    "neutral": 1.0,
    "angry": 1.2,
    "tired_low": 1.2,
    "sad": 2.0,
    "anxious": 2.0,
    "lonely": 2.0,
    "comfort_need": 2.0,
}


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").replace("\r", "\n").split()).strip()


def normalize_label(label: Any) -> str:
    value = str(label or "").strip()
    return LABEL_ALIASES.get(value, value or "neutral")


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_user_text(messages: Any) -> Optional[str]:
    if not isinstance(messages, list):
        return None
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return normalize_text(msg.get("content"))
    return None


def extract_label(obj: Dict[str, Any]) -> str:
    meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
    for key in ("emotion_label", "emotion8", "emotion", "label"):
        value = obj.get(key)
        if value is None:
            value = meta.get(key)
        if value is not None:
            return normalize_label(value)
    return "neutral"


def load_text_label_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for idx, obj in enumerate(iter_jsonl(path), start=1):
        text = normalize_text(obj.get("text"))
        if not text:
            text = extract_user_text(obj.get("messages"))
        if not text:
            continue
        label = extract_label(obj)
        rows.append(
            {
                "id": obj.get("id", idx),
                "text": text,
                "label": label,
                "source": str(path),
            }
        )
    return rows
