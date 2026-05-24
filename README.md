# Fast-dLLM v2 Compute-Skipping Experiments

Implementation and evaluation of two compute-skipping policies for Fast-dLLM v2 (Qwen2.5-7B-Instruct fine-tune), evaluated on GSM8K.

This repository contains:
- **Phase A**: Motivation analysis — 11 figures characterizing cross-step similarity and attention sparsity in the model.
- **Phase B**: Compute-skipping policies — 4 `SkipPolicy` classes covering token-level (cosine threshold, top-k%) and layer-level (cosine threshold with avg/max aggregation) skipping, swept over 18 settings × 100 GSM8K samples.

## Headline result

Token-level **TopK with k=50** achieves **85% GSM8K accuracy** (vs. 80% baseline) with **49.1% FLOPs reduction**, by always preserving the most-changed 50% of tokens per step while reusing the rest. See [writeup_phase_b.md](writeup_phase_b.md) for the full analysis.

![Phase B Accuracy vs FLOPs](figs/phase_b_acc_flops_v2.png)

## Setup

```bash
conda create -n fastdllm python=3.11
conda activate fastdllm
pip install -r requirements.txt
```

Download the Fast-dLLM v2 7B model from HuggingFace:

```bash
huggingface-cli download Efficient-Large-Model/Fast_dLLM_v2_7B \
    --local-dir models/Fast_dLLM_v2_7B
```

Update the `MODEL_PATH` constant at the top of the driver scripts to point to your local model directory.

## Reproducing results

### Phase A: motivation figures

```bash
# Run 100 GSM8K samples and log per (step, layer) similarity data
python src/run_motivation_100.py

# Generate figures
python src/plot_token_motivation_100.py
python src/plot_layer_motivation_100.py
python src/plot_attn_weight_histogram.py
```

Outputs go to `figs/`.

### Phase B: 18-setting compute-skipping sweep

```bash
# Run all 17 non-baseline settings × 100 samples (~14 hours on RTX 5080)
python src/run_skip_experiment.py

# Aggregate results and generate Figure 1 + Table 1
python src/aggregate_results.py
python src/improve_phase_b_plot.py
```

To run a single setting:

```bash
python src/run_skip_experiment.py --setting token_topk_50
```

### Sanity tests

The `sanity/` folder contains short scripts verifying:
- `NoSkipPolicy` reproduces the baseline forward exactly
- `TokenTopKPolicy`, `TokenCossimPolicy`, `LayerCossimPolicy` actually substitute outputs and yield expected reuse statistics

## Repository layout

| Path | Purpose |
|---|---|
| `src/policies.py` | 4 `SkipPolicy` classes + `ALL_SETTINGS` list (18 configs) |
| `src/step_cache.py` | Manager + hooks; captures similarities and substitutes attn/MLP outputs based on skip mask |
| `src/run_skip_experiment.py` | Phase B driver: 17 settings × 100 GSM8K samples with resume support |
| `src/run_motivation_100.py` | Phase A driver: 100 samples baseline + per (step, layer) similarity dump |
| `src/aggregate_results.py` | Compute FLOPs reduction and Pareto frontier from per-sample dumps |
| `src/improve_phase_b_plot.py` | Render Figure 1 (acc-FLOPs scatter) and Table 1 |
| `writeup_phase_b.md` | 2-page summary of implementation, formula, key findings, and future work |
| `results/phase_b_aggregate.csv` | Per-setting accuracy, reuse rate, FLOPs reduction |
| `results/summary_per_setting/` | 17 individual setting summaries (JSON) |
| `figs/` | Phase A motivation figures (16) + Phase B Figure 1 + Table 1 |

## Implementation notes

We use **PyTorch forward hooks** in `step_cache.py` to implement skipping, rather than directly modifying `modeling.py`. This decouples the skipping logic from the model architecture and keeps the codebase cleaner. The single change to `modeling.py` is a `_compute_attention_stats` helper used only for Phase A's attention-weight histogram. See `writeup_phase_b.md` §1 for the rationale.

## Key findings

1. **Self-reinforcing token-level reuse**: Once a token is reused, its hidden state is bitwise-identical next step, so its cross-step similarity becomes exactly 1.0 — triggering reuse again forever. All five `token_cossim` thresholds (0.96–0.995) collapse to the same behavior. This is a fundamental limitation of self-referential decision criteria.
2. **Layer-level avg yields a clean Pareto curve**, with the 0.995 threshold as a sweet spot (85% acc, 5% FLOPs reduction).
3. **Layer-level max is unusable at block_size=32**: `max(sim)` over 32 tokens is almost always ≈ 1.0, triggering whole-layer skip on every step. All 5 max thresholds collapse to 0% accuracy.
4. **TopK (especially k=50) is the practical winner**: by guaranteeing computation on the most-changed tokens, it sidesteps the self-reinforcing loop and achieves the best (acc, FLOPs) trade-off.

## Hardware used

- GPU: NVIDIA RTX 5080 (16 GB VRAM)
- Python: 3.11
- PyTorch: 2.11.0 + CUDA 12.8 (sm_120)
- Total compute time for Phase B sweep: ~13 hours

## License

MIT (see [LICENSE](LICENSE)).

## Citation

If you use this code or findings, please cite:

```bibtex
@misc{fastdllm-skipping-2026,
  author = {Your Name},
  title  = {Fast-dLLM v2 Compute-Skipping Experiments},
  year   = {2026},
  url    = {https://github.com/your-username/fastdllm-skipping}
}
```
