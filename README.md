# BehaviorDrop

BehaviorDrop is a behavior-guided mixed-precision quantization pipeline for emotionally supportive companion LLMs. It selects high-precision MLP layers by measuring output-level behavior degradation caused by temporary INT4 perturbation.

This package contains the core code and small data files needed to reproduce the main BehaviorDrop workflow.

## Directory

```text
BehaviorDrop/
  train.py                         # EQ-DoRA / LoRA / DoRA companion SFT training
  infer.py                         # PyTorch-side fake INT4/INT8/mixed-precision inference
  compute_behavior_drop.py          # Layer-wise BehaviorDrop sensitivity scoring
  compute_eaas.py                   # EAAS gradient-saliency baseline
  build_ebs_allocation.py           # EAAS/BehaviorDrop fusion and protected-layer export
  common.py                         # Shared labels, JSONL utilities, normalization

  data/
    train.jsonl                     # Companion SFT training data
    eval.jsonl                      # Original 256-prompt EAQ evaluation set
    eaq_ebs_probe.jsonl             # 64-prompt probing split for layer selection
    eaq_ebs_test.jsonl              # 192-prompt held-out EAQ-Test split
    eaq_ebs_split_report.json       # Probe/test split report
    fp16_ref_response.jsonl         # FP16 reference responses for EAQ prompts

  eval/
    evaluate_response_deployment.py # TTS/format behavior rules
    evaluate_emotion_match.py       # Rule-based emotion-response matching
    evaluate_behavior_alignment.py  # FP16-relative behavior retention
    collect_random_layercount_results.py
```

## Environment

Install dependencies with:

```bash
pip install -r requirements.txt
```

Recommended hardware: one CUDA GPU with enough memory for Qwen2.5-1.5B-Instruct. CPU execution is possible for small smoke tests but very slow.

## Data Splits

The main evaluation protocol uses disjoint splits:

- `data/eaq_ebs_probe.jsonl`: used only for BehaviorDrop / EAAS / EBS layer ranking.
- `data/eaq_ebs_test.jsonl`: used only for final reported EAQ-Test metrics.
- `data/eaq_ebs_split_report.json`: records the split seed and label counts.

Do not use `data/eaq_ebs_test.jsonl` to select protected layers.

## 1. Train EQ-DoRA Reference Model

Download or place the base model, for example Qwen2.5-1.5B-Instruct, then run:

```bash
python train.py \
  --model /path/to/Qwen2.5-1.5B-Instruct \
  --data data/train.jsonl \
  --out checkpoints/eq_dora_r8 \
  --peft_method eq_dora \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --epochs 3 \
  --lr 2e-4 \
  --batch 2 \
  --grad_accum 8 \
  --seed 42 \
  --bf16
```

Merge the adapter into the base model with your preferred PEFT merge script, or use an already merged model directory. In the commands below, this directory is called:

```text
/path/to/merged_eq_dora_model
```

## 2. Generate FP16 Reference Responses

The package includes `data/fp16_ref_response.jsonl` for convenience. To regenerate it, run inference without quantization using your own generation script or adapt `infer.py` with a no-quantization path. The reference file must contain `id`, `text`, `label`, and `response` fields.

## 3. Compute BehaviorDrop Layer Scores

```bash
python compute_behavior_drop.py \
  --model /path/to/merged_eq_dora_model \
  --input data/eaq_ebs_probe.jsonl \
  --ref data/fp16_ref_response.jsonl \
  --out_csv outputs/behavior_drop_layer_scores.csv \
  --out_json outputs/behavior_drop_layer_scores.json \
  --module_type mlp \
  --layers all \
  --bits 4 \
  --group_size 128 \
  --limit 64 \
  --bf16
```

The output CSV contains `behavior_drop_score` and `normalized_behavior_drop` for each probed layer.

## 4. Compute EAAS Baseline Scores

```bash
python compute_eaas.py \
  --model /path/to/merged_eq_dora_model \
  --input data/eaq_ebs_probe.jsonl \
  --out_dir outputs/eaas \
  --max_samples 32 \
  --bf16
```

This produces `outputs/eaas/eaas_layer_scores.csv`.

## 5. Build Protected-Layer Lists

For EBS-fusion ablations:

```bash
python build_ebs_allocation.py \
  --eaas outputs/eaas/eaas_layer_scores.csv \
  --behavior outputs/behavior_drop_layer_scores.csv \
  --out_dir outputs/allocation \
  --budgets 4.75
```

For pure BehaviorDrop k=7, use the top seven layers from `behavior_drop_layer_scores.csv` sorted by `behavior_drop_score`, and save them as a comma-separated file, for example:

```text
outputs/allocation/behaviordrop_k7_layers.txt
```

The paper setting uses seven protected MLP layers.

## 6. Run W4/W8 Fake-Quantized Inference on EAQ-Test

```bash
python infer.py \
  --model /path/to/merged_eq_dora_model \
  --input data/eaq_ebs_test.jsonl \
  --output outputs/behaviordrop_k7_response.jsonl \
  --quant_report outputs/behaviordrop_k7_quant_report.json \
  --strategy custom_mlp_int8 \
  --protected_layers outputs/allocation/behaviordrop_k7_layers.txt \
  --group_size 128 \
  --bf16
```

For internal baselines:

```bash
python infer.py --model /path/to/merged_eq_dora_model --input data/eaq_ebs_test.jsonl --output outputs/uniform_int4_response.jsonl --quant_report outputs/uniform_int4_quant_report.json --strategy uniform_int4 --group_size 128 --bf16
python infer.py --model /path/to/merged_eq_dora_model --input data/eaq_ebs_test.jsonl --output outputs/uniform_int8_response.jsonl --quant_report outputs/uniform_int8_quant_report.json --strategy uniform_int8 --group_size 128 --bf16
```

## 7. Evaluate Behavior Metrics

```bash
python eval/evaluate_response_deployment.py \
  --pred outputs/behaviordrop_k7_response.jsonl \
  --report outputs/behaviordrop_k7_response_report.json \
  --csv outputs/behaviordrop_k7_response_summary.csv

python eval/evaluate_emotion_match.py \
  --pred outputs/behaviordrop_k7_response.jsonl \
  --report outputs/behaviordrop_k7_emotion_match_report.json \
  --csv outputs/behaviordrop_k7_emotion_match_summary.csv

python eval/evaluate_behavior_alignment.py \
  --ref data/fp16_ref_response.jsonl \
  --pred outputs/behaviordrop_k7_response.jsonl \
  --report outputs/behaviordrop_k7_behavior_alignment_report.json \
  --csv outputs/behaviordrop_k7_behavior_alignment_summary.csv
```

## 8. Paper Result Summary

The lightweight result files under `result/` summarize the main paper evidence:

- `result/tables/signal_ablation_summary.csv` records EAQ-Test behavior metrics for FP16, RTN, EAAS-only, EBS-fusion, and BehaviorDrop policies.
- `result/tables/quant_baseline_summary.csv` records the GPTQ and AWQ INT4 baseline results.
- `result/tables/random_layercount_summary.csv` records exact-layer-count random controls.
- `result/tables/efficiency_cost_summary.csv` records the average bit budget, estimated linear size, compression ratio, and throughput summary.
- `result/layer_scores/behavior_drop_layer_scores.csv` and `result/layer_scores/eaas_layer_scores.csv` provide the layer-ranking signals used in the diagnostic analysis.

In the reported EAQ-Test setting, BehaviorDrop k=7 improves negative-emotion matching over AWQ INT4 from 0.23 to 0.66 and reduces bad-format rate from 0.30 to 0.11, while using a 4.75-bit W4/W8 mixed-precision policy. The result files are provided so readers can inspect the table values without rerunning the full pipeline.
## Notes on Exact Reproduction

Exact numerical reproduction requires the same base model, merged EQ-DoRA checkpoint, decoding configuration, package versions, and GPU/runtime settings. The included `data/fp16_ref_response.jsonl` can be used to reproduce the layer-scoring and evaluation protocol once a compatible merged model is available.


