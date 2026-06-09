#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import inspect
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset, random_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)


class MessagesDataset(Dataset):
    def __init__(self, path: str, max_samples: int = 0):
        self.items: List[Dict[str, Any]] = []
        self.label_counts = Counter()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                messages = obj.get("messages")
                if not isinstance(messages, list) or len(messages) < 2:
                    continue
                if messages[-1].get("role") != "assistant":
                    continue
                emotion_label = extract_emotion_label(obj)
                self.items.append({"messages": messages, "emotion_label": emotion_label})
                self.label_counts[emotion_label] += 1
                if max_samples > 0 and len(self.items) >= max_samples:
                    break

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


EMOTION_WEIGHTS = {
    "neutral": 1.0,
    "happy": 1.0,
    "tired_low": 1.2,
    "angry": 1.2,
    "sad": 1.4,
    "anxious": 1.4,
    "lonely": 1.4,
    "comfort_need": 1.5,
}

EMOTION_LABEL_ALIASES = {
    "neutral": "neutral",
    "中性/日常": "neutral",
    "happy": "happy",
    "开心/愉悦": "happy",
    "tired_low": "tired_low",
    "burnout": "tired_low",
    "低落/疲惫": "tired_low",
    "angry": "angry",
    "生气/烦躁": "angry",
    "sad": "sad",
    "sadness": "sad",
    "悲伤/难过": "sad",
    "anxious": "anxious",
    "fear": "anxious",
    "overwhelmed": "anxious",
    "焦虑/压力": "anxious",
    "lonely": "lonely",
    "孤独/需要陪伴": "lonely",
    "comfort_need": "comfort_need",
    "calm": "comfort_need",
    "安慰/放松需求": "comfort_need",
}


def extract_emotion_label(obj: Dict[str, Any]) -> str:
    meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
    for key in ("emotion_label", "emotion8", "emotion", "label"):
        value = obj.get(key)
        if value is None:
            value = meta.get(key)
        if value is not None:
            label = str(value).strip()
            if label:
                return EMOTION_LABEL_ALIASES.get(label, label)
    return "neutral"


def parse_emotion_weights(raw: str) -> Dict[str, float]:
    if not raw:
        return dict(EMOTION_WEIGHTS)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("--emotion_weights must be a JSON object, e.g. '{\"sad\":1.4}'") from exc
    if not isinstance(obj, dict):
        raise ValueError("--emotion_weights must be a JSON object")

    weights = dict(EMOTION_WEIGHTS)
    for key, value in obj.items():
        norm_key = EMOTION_LABEL_ALIASES.get(str(key).strip(), str(key).strip())
        weights[norm_key] = float(value)
    return weights


@dataclass
class AssistantOnlyCollator:
    tokenizer: Any
    max_seq_len: int = 512
    emotion_weights: Optional[Dict[str, float]] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        input_ids_list = []
        attn_list = []
        labels_list = []
        weight_list = []

        for feat in features:
            messages = feat["messages"]
            prompt_messages = messages[:-1]

            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            prompt_ids = self.tokenizer(
                prompt_text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_seq_len,
            )["input_ids"]
            full = self.tokenizer(
                full_text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_seq_len,
                return_attention_mask=True,
            )

            input_ids = torch.tensor(full["input_ids"], dtype=torch.long)
            attention_mask = torch.tensor(full["attention_mask"], dtype=torch.long)
            labels = input_ids.clone()
            prompt_len = min(len(prompt_ids), labels.shape[0])
            labels[:prompt_len] = -100
            labels[labels == pad_id] = -100

            input_ids_list.append(input_ids)
            attn_list.append(attention_mask)
            labels_list.append(labels)
            if self.emotion_weights is not None:
                label = feat.get("emotion_label", "neutral")
                weight_list.append(float(self.emotion_weights.get(label, 1.0)))

        batch = {
            "input_ids": self._pad(input_ids_list, pad_id),
            "attention_mask": self._pad(attn_list, 0),
            "labels": self._pad(labels_list, -100),
        }
        if weight_list:
            batch["emotion_weights"] = torch.tensor(weight_list, dtype=torch.float32)
        return batch

    @staticmethod
    def _pad(tensors: List[torch.Tensor], pad_value: int) -> torch.Tensor:
        max_len = max(t.shape[0] for t in tensors)
        padded = []
        for tensor in tensors:
            if tensor.shape[0] < max_len:
                pad = torch.full((max_len - tensor.shape[0],), pad_value, dtype=tensor.dtype)
                tensor = torch.cat([tensor, pad], dim=0)
            padded.append(tensor)
        return torch.stack(padded, dim=0)


class SampleCallback(TrainerCallback):
    def __init__(self, tokenizer, sample_user: str, every_steps: int, max_new_tokens: int):
        self.tokenizer = tokenizer
        self.sample_user = sample_user
        self.every_steps = every_steps
        self.max_new_tokens = max_new_tokens

    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if step <= 0 or step % self.every_steps != 0:
            return

        model = kwargs.get("model")
        if model is None:
            return

        messages = [
            {
                "role": "system",
                "content": "你是一个中文情感陪伴机器人。请自然、真诚、温和地回应用户，不要输出动作标签。",
            },
            {"role": "user", "content": self.sample_user},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)

        model.eval()
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        print("\n" + "=" * 24 + f" SAMPLE step={step} " + "=" * 24)
        print(text)
        print("=" * 70 + "\n")
        model.train()


class EmotionWeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        emotion_weights = inputs.pop("emotion_weights", None)
        if emotion_weights is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        vocab_size = shift_logits.shape[-1]
        token_loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shift_labels.shape)

        valid_tokens = shift_labels.ne(-100)
        per_sample_loss = token_loss.sum(dim=1) / valid_tokens.sum(dim=1).clamp_min(1)
        weighted_loss = (per_sample_loss * emotion_weights.to(per_sample_loss.device)).mean()

        return (weighted_loss, outputs) if return_outputs else weighted_loss


def build_training_args(args) -> TrainingArguments:
    sig = inspect.signature(TrainingArguments.__init__)
    supported = set(sig.parameters.keys())
    kwargs = dict(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=2,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataloader_num_workers=args.num_workers,
        max_grad_norm=args.max_grad_norm,
    )
    if "optim" in supported:
        kwargs["optim"] = "adamw_torch"
    if "evaluation_strategy" in supported:
        kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in supported:
        kwargs["eval_strategy"] = "steps"
    kwargs = {k: v for k, v in kwargs.items() if k in supported}
    return TrainingArguments(**kwargs)


def parse_args():
    ap = argparse.ArgumentParser(description="Train a compact Qwen companion-chat LoRA model.")
    ap.add_argument("--model", required=True, help="base model, e.g. /media/dell/SDD1/models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", required=True, help="companion messages jsonl")
    ap.add_argument("--out", required=True)

    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--lr_scheduler", type=str, default="cosine")
    ap.add_argument("--max_grad_norm", type=float, default=0.3)

    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--qlora", action="store_true")
    ap.add_argument("--local_files_only", action="store_true")

    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--eval_ratio", type=float, default=0.05)
    ap.add_argument("--eval_steps", type=int, default=100)
    ap.add_argument("--num_workers", type=int, default=2)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample_steps", type=int, default=50)
    ap.add_argument("--sample_user", default="我最近压力很大，感觉自己什么都做不好。")
    ap.add_argument("--sample_max_new_tokens", type=int, default=96)

    ap.add_argument("--peft_method", choices=["lora", "dora", "eq_dora"], default="lora")
    ap.add_argument(
        "--emotion_weights",
        default=json.dumps(EMOTION_WEIGHTS, ensure_ascii=False),
        help="JSON object mapping emotion labels to CE weights; used by --peft_method eq_dora.",
    )
    ap.add_argument("--max_train_samples", type=int, default=0, help="0 means use all; useful for smoke tests")
    ap.add_argument("--max_eval_samples", type=int, default=0, help="0 means keep the eval split unchanged")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    return ap.parse_args()


def resolve_model_path(model: str, local_files_only: bool) -> str:
    path = Path(model).expanduser()
    looks_like_path = model.startswith("/") or model.startswith(".") or os.sep in model
    if path.exists():
        required = ["config.json", "tokenizer_config.json"]
        missing = [name for name in required if not (path / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"Model directory exists but is missing {missing}: {path}\n"
                "Please use a complete Hugging Face model directory."
            )
        return str(path)

    if local_files_only or looks_like_path:
        parent = path.parent if str(path.parent) else Path(".")
        nearby = []
        if parent.exists():
            nearby = sorted(p.name for p in parent.glob("*Qwen*") if p.is_dir())
        hint = f"\nAvailable Qwen-like dirs under {parent}: {nearby}" if nearby else ""
        raise FileNotFoundError(
            f"Local model path does not exist: {path}\n"
            "Download Qwen2.5-0.5B-Instruct first, or pass an existing local model path."
            f"{hint}"
        )

    return model


def main():
    args = parse_args()
    args.model = resolve_model_path(args.model, args.local_files_only)

    if args.bf16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "Current CUDA device does not report bf16 support. "
            "Please remove --bf16 and use --fp16, or train in fp32 for a stability check."
        )

    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'peft'. Install it in your training environment, e.g. `pip install peft`."
        ) from exc

    set_seed(args.seed)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    if args.local_files_only:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = None
    if args.qlora:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    load_kwargs = dict(
        pretrained_model_name_or_path=args.model,
        trust_remote_code=True,
        device_map="auto",
        local_files_only=args.local_files_only,
        quantization_config=quant_cfg,
    )
    if dtype is not None and not args.qlora:
        load_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    model.config.use_cache = False

    if args.qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()

    lora_kwargs = dict(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    if args.peft_method in ("dora", "eq_dora"):
        if "use_dora" not in inspect.signature(LoraConfig.__init__).parameters:
            raise RuntimeError(
                "Current peft version does not support DoRA (`LoraConfig(use_dora=True)`). "
                "Please upgrade peft in the training environment."
            )
        lora_kwargs["use_dora"] = True
    lora_cfg = LoraConfig(**lora_kwargs)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds = MessagesDataset(args.data, max_samples=args.max_train_samples)
    if len(ds) < 2:
        raise ValueError(f"dataset too small: {len(ds)}")
    n_eval = max(1, int(len(ds) * args.eval_ratio)) if args.eval_ratio > 0 else 0
    n_train = len(ds) - n_eval
    if n_eval > 0:
        train_ds, eval_ds = random_split(ds, [n_train, n_eval])
    else:
        train_ds, eval_ds = ds, None
    if args.max_eval_samples > 0 and eval_ds is not None:
        eval_ds = torch.utils.data.Subset(eval_ds, range(min(args.max_eval_samples, len(eval_ds))))
    print(f"[INFO] dataset train={len(train_ds)} eval={len(eval_ds) if eval_ds else 0}")
    print(f"[INFO] raw emotion labels={dict(ds.label_counts)}")

    emotion_weights = parse_emotion_weights(args.emotion_weights)
    if args.peft_method == "eq_dora":
        print(f"[INFO] EQ-DoRA emotion weights={emotion_weights}")
    else:
        emotion_weights = None

    callbacks = []
    if args.sample_steps > 0:
        callbacks.append(
            SampleCallback(
                tokenizer=tokenizer,
                sample_user=args.sample_user,
                every_steps=args.sample_steps,
                max_new_tokens=args.sample_max_new_tokens,
            )
        )

    trainer_cls = EmotionWeightedTrainer if args.peft_method == "eq_dora" else Trainer
    trainer = trainer_cls(
        model=model,
        args=build_training_args(args),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=AssistantOnlyCollator(
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            emotion_weights=emotion_weights,
        ),
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    run_config = {
        "peft_method": args.peft_method,
        "emotion_weights": parse_emotion_weights(args.emotion_weights),
        "emotion_label_counts": dict(ds.label_counts),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
    }
    Path(args.out, "paper_train_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[DONE] saved {args.peft_method} adapter to {args.out}")


if __name__ == "__main__":
    main()
