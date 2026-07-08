# BehaviorDrop key results

This folder contains lightweight result files for reproducing the paper tables and figures without committing the full `outputs_paper/` directory.

## Contents

- `tables/`: CSV summaries used by the main result table, ablation table, random controls, GPTQ/AWQ baselines, and compression-cost table.
- `layer_scores/`: BehaviorDrop, EBS-fusion, and EAAS layer-score files used for layer ranking and diagnostics.
- `figures/`: PNG versions of paper-related generated figures.
- `reports/`: selected quantization reports for BehaviorDrop, RTN, GPTQ, and AWQ settings.
- `data_split/`: split report documenting the probing/test separation.

Large checkpoints, exported model weights, full generated-response dumps, and the full `outputs_paper/` tree are intentionally excluded.

## Main files

- `tables/signal_ablation_summary.csv`: EAQ-Test results for FP16, RTN, EAAS-only, EBS-fusion, and BehaviorDrop policies.
- `tables/quant_baseline_summary.csv`: GPTQ/AWQ INT4 baseline results.
- `tables/random_layercount_summary.csv`: exact-layer-count random controls.
- `tables/efficiency_cost_summary.csv`: bit budget, linear size, compression ratio, and throughput summary.
- `layer_scores/behavior_drop_layer_scores.csv`: layer-wise BehaviorDrop scores.
- `layer_scores/eaas_layer_scores.csv`: layer-wise EAAS saliency scores.
