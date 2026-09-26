# TIGER Generative Recommendation on Apple Silicon

A resource-constrained reproduction and extension of TIGER-style generative recommendation with **RQ-VAE Semantic IDs** and a Transformer retrieval model. This repository is based on the PyTorch RQ-VAE / generative-retrieval implementation by **Edoardo Botta** (MIT License) and adds Apple Silicon support, reproducible evaluation utilities, training fixes, and a hierarchical Semantic-ID loss experiment.

> This is not an exact reproduction of the original paper's hardware/training protocol. The experiments below were run on a MacBook Air with Apple M4 and 24 GB unified memory using PyTorch MPS.

## Pipeline

1. Item content embeddings are encoded and quantized by a multi-level RQ-VAE.
2. Each item is represented by a hierarchical Semantic ID `(RQ1, RQ2, RQ3, DEDUP)`.
3. User interaction sequences are converted to Semantic-ID sequences.
4. A Transformer autoregressively generates the next item's Semantic ID.
5. Prefix verification restricts generation to Semantic IDs that exist in the item corpus.

## What this fork adds

- **Apple Silicon / MPS compatibility**
  - Pure-PyTorch fallback when Triton is unavailable.
  - Padded `nn.Transformer` path for MPS, avoiding unsupported jagged/nested attention during training.
  - MPS-compatible generation and evaluation workflow.
- **Training reliability**
  - Reproducible Python/PyTorch seeding.
  - Correct resume iteration range and final-checkpoint saving after resume.
  - Padded-path unreduced loss handling.
- **Evaluation / analysis utilities**
  - Full Semantic-ID HitRate@1/5/10 subset evaluator.
  - Position-wise Semantic-ID loss/accuracy analysis on train or evaluation data.
  - MPS generation smoke test.
- **HSWL: Hierarchical Semantic-ID Weighted Loss**
  - Configurable loss weights for the four Semantic-ID positions.
  - Controlled experiment against an equal-weight baseline.

## HSWL

The standard objective gives each Semantic-ID position equal weight:

```text
Baseline = [1.00, 1.00, 1.00, 1.00]
             RQ1   RQ2   RQ3  DEDUP
```

The tested HSWL-A configuration reallocates the same total weight toward earlier hierarchy levels:

```text
HSWL-A   = [1.50, 1.25, 0.75, 0.50]
```

For per-example position losses `L_d`, the decoder objective is:

```text
L = mean_batch( sum_d w_d * L_d )
```

The weights sum to 4 in both configurations to reduce loss-scale / learning-rate confounding.

## Experimental setup

| Setting | Value |
|---|---|
| Dataset | Amazon Reviews - Beauty |
| Hardware | Apple M4, 24 GB unified memory |
| Backend | PyTorch MPS |
| RQ-VAE | Official pretrained Amazon Beauty high-entropy checkpoint |
| Semantic-ID dimensions | RQ1, RQ2, RQ3, DEDUP |
| Codebook size | 256 |
| Decoder batch size | 256 |
| Learning rate | 3e-4 |
| Weight decay | 0.035 |
| Decoder | 8 attention layers, 8 heads, 512 attention dim |
| Dropout | 0.3 |
| Controlled 5k seed | 2026 |

The official pretrained RQ-VAE is used for the decoder experiments. The local RQ-VAE smoke run is only an engineering check and is not used as the tokenizer for the reported results.

## Results

### Decoder training-budget study

Full evaluation set (`22,363` examples), exact match over all four Semantic-ID positions:

| Decoder checkpoint | H@1 | H@5 | H@10 |
|---:|---:|---:|---:|
| 10k steps | 0.002325 | 0.008943 | **0.016858** |
| 20k steps | 0.002102 | 0.007065 | 0.016500 |
| 30k steps | 0.002325 | 0.007557 | 0.016500 |

Training loss continued to decrease across these checkpoints, but the evaluated full-ID HitRate did not improve monotonically. These numbers are reported as a training-budget study, not as validation-selected best checkpoints.

### HSWL controlled 5k experiment

Same training budget, seed, model, tokenizer and optimizer settings; only the Semantic-ID loss weights differ. Position-wise accuracies are measured on a fixed 4,096-example training subset. End-to-end H@K is measured on the same fixed 1,024-example evaluation subset for both models.

| Model | RQ1 Acc | RQ2 Acc | RQ3 Acc | DEDUP Acc | H@5 | H@10 |
|---|---:|---:|---:|---:|---:|---:|
| Equal-weight baseline | 3.760% | 10.327% | 13.159% | 92.725% | 0.001953 | 0.002930 |
| HSWL-A | 3.760% | **13.257%** | **14.282%** | 92.749% | 0.001953 | 0.002930 |

**Finding:** HSWL-A changed the optimization behavior and improved RQ2/RQ3 teacher-forced token accuracy at 5k steps, but the improvement did **not** translate into an observed end-to-end HitRate gain on the 1,024-example evaluation subset. This is intentionally reported as a negative end-to-end result rather than as a recommendation-quality improvement.

## Installation

```bash
pip install -r requirements.txt
```

The original data pipeline downloads/processes supported datasets through the repository code. For the Amazon Beauty experiments, place the pretrained RQ-VAE checkpoint at:

```text
trained_models/rqvae_amazon_beauty/checkpoint_high_entropy.pt
```

The upstream pretrained Amazon Beauty RQ-VAE is available from the original author's Hugging Face repository: https://huggingface.co/edobotta/rqvae-amazon-beauty

On Apple Silicon, commands that may encounter unsupported MPS operators can be run with:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 <command>
```

## Reproducing the main experiments

### 5k equal-weight baseline

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 MPS_LAUNCH_BLOCKING=1 \
python train_decoder.py configs/decoder_amazon_baseline_5k_seed2026.gin
```

### 5k HSWL-A

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 MPS_LAUNCH_BLOCKING=1 \
python train_decoder.py configs/decoder_amazon_hswl_a_5k_seed2026.gin
```

### End-to-end Semantic-ID evaluation

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python eval_decoder_subset_mps.py \
  --checkpoint out/decoder/amazon_baseline_5k_seed2026/checkpoint_4999.pt \
  --num-samples 1024 \
  --batch-size 2
```

### Position-wise training analysis

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python analyze_train_token_loss_mps.py \
  --checkpoint out/decoder/amazon_baseline_5k_seed2026/checkpoint_4999.pt \
  --num-samples 4096 \
  --batch-size 256
```

For full evaluation-position analysis, use `analyze_token_loss_mps.py`.

## Important implementation notes

- `model_jagged_mode=False` is used for the reported MPS decoder experiments because the original jagged/nested attention path is not fully supported on MPS.
- `ops/triton/jagged.py` keeps the Triton path for CUDA when available and provides a pure-PyTorch fallback otherwise.
- Generation retains the original stochastic candidate selection with `torch.multinomial`; experimental deterministic candidate generation was not retained because it produced no HitRate change in screening.
- HSWL defaults to equal weights, so existing decoder checkpoints remain compatible.
- Large datasets, experiment outputs and model checkpoints are intentionally excluded from version control.

## Repository layout

```text
configs/                     experiment configurations
modules/                     RQ-VAE, Transformer and retrieval model
ops/triton/jagged.py         Triton path + PyTorch MPS/CPU fallback
data/                        dataset processing
train_rqvae.py               RQ-VAE training
train_decoder.py             generative retrieval training
eval_decoder_subset_mps.py   full-ID HitRate evaluation
analyze_token_loss_mps.py    evaluation position analysis
analyze_train_token_loss_mps.py
                             training-subset position analysis
test_generation_mps.py       generation smoke test
```

## Limitations

- The reported experiments are a resource-constrained Apple Silicon reproduction, not an exact reproduction of the paper's accelerator setup or full training budget.
- Decoder experiments use the upstream pretrained RQ-VAE rather than a locally trained full RQ-VAE.
- The 5k HSWL comparison uses one controlled seed; it should not be interpreted as a multi-seed statistical result.
- The 1,024-example H@K comparison is a screening subset. The full 22,363-example numbers above are reported only for the 10k/20k/30k baseline checkpoints.
- HSWL improves intermediate token prediction in the reported experiment but does not establish an end-to-end recommendation improvement.

## Upstream and references

This project is derived from Edoardo Botta's MIT-licensed RQ-VAE generative-retrieval implementation. The original copyright notice is preserved in `LICENSE`.

- Upstream implementation: https://github.com/EdoardoBotta/RQ-VAE
- Rajput et al., *Recommender Systems with Generative Retrieval*: https://arxiv.org/abs/2305.05065
- Jang et al., *Categorical Reparameterization with Gumbel-Softmax*: https://openreview.net/forum?id=rkE3y85ee
- Fifty et al., *Restructuring Vector Quantization with the Rotation Trick*: https://arxiv.org/abs/2410.06424
