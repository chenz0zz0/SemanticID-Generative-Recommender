# SemanticID Generative Recommender

A generative recommendation system that represents items with **RQ-VAE Semantic IDs** and autoregressively generates the next item's ID with a Transformer. This repository also includes **Hierarchical Semantic-ID Weighted Loss (HSWL)** experiments and an Apple Silicon / PyTorch MPS execution path.

The project is developed from Edoardo Botta's MIT-licensed PyTorch RQ-VAE / generative-retrieval implementation. The original copyright notice is preserved in `LICENSE`; the extensions in this repository focus on MPS compatibility, training reliability, reproducible evaluation, and hierarchical Semantic-ID modeling.

## Method

```text
Item content embedding
        ↓
      RQ-VAE
        ↓
(RQ1, RQ2, RQ3, DEDUP)
        ↓
User Semantic-ID sequence
        ↓
    Transformer
        ↓
Next-item Semantic-ID generation
```

The pipeline first maps each item to a hierarchical Semantic ID. A Transformer then models user interaction histories as Semantic-ID sequences and generates the next item's ID autoregressively. Prefix verification restricts generated IDs to items that exist in the corpus.

## Project extensions

- **Hierarchical Semantic-ID Weighted Loss (HSWL):** configurable loss weights for RQ1, RQ2, RQ3 and DEDUP, with controlled comparison against equal weighting.
- **Apple Silicon / MPS support:** pure-PyTorch fallback when Triton is unavailable and a padded `nn.Transformer` path for unsupported jagged/nested MPS attention.
- **Training reliability:** reproducible Python/PyTorch seeding, corrected resume iteration range, final-checkpoint saving after resume, and padded-path unreduced-loss handling.
- **Evaluation utilities:** full Semantic-ID HitRate@1/5/10 subset evaluation, position-wise token loss/accuracy analysis, and an MPS generation smoke test.

## HSWL

The equal-weight decoder objective uses:

```text
Baseline = [1.00, 1.00, 1.00, 1.00]
             RQ1   RQ2   RQ3  DEDUP
```

The tested HSWL-A configuration reallocates the same total weight toward earlier hierarchy levels:

```text
HSWL-A   = [1.50, 1.25, 0.75, 0.50]
```

For per-example position losses `L_d`:

```text
L = mean_batch(sum_d w_d * L_d)
```

Both configurations sum to 4, reducing loss-scale / learning-rate confounding in the controlled comparison.

## Experimental setup

| Setting | Value |
|---|---|
| Dataset | Amazon Reviews - Beauty |
| Hardware | Apple M4, 24 GB unified memory |
| Backend | PyTorch MPS |
| RQ-VAE | Upstream pretrained Amazon Beauty high-entropy checkpoint |
| Semantic-ID dimensions | RQ1, RQ2, RQ3, DEDUP |
| Codebook size | 256 |
| Decoder batch size | 256 |
| Learning rate | 3e-4 |
| Weight decay | 0.035 |
| Decoder | 8 attention layers, 8 heads, 512 attention dim |
| Dropout | 0.3 |
| Controlled 5k seed | 2026 |

The pretrained RQ-VAE is used for the reported decoder experiments. The local RQ-VAE smoke run is only an engineering check and is not used as the tokenizer for the reported results.

## Results

### Decoder training-budget study

Full evaluation set (`22,363` examples), exact match over all four Semantic-ID positions:

| Decoder checkpoint | H@1 | H@5 | H@10 |
|---:|---:|---:|---:|
| 10k steps | 0.002325 | 0.008943 | **0.016858** |
| 20k steps | 0.002102 | 0.007065 | 0.016500 |
| 30k steps | 0.002325 | 0.007557 | 0.016500 |

Training loss continued to decrease across these checkpoints, while evaluated full-ID HitRate did not improve monotonically. These results are a training-budget study rather than validation-selected best checkpoints.

### HSWL controlled 5k experiment

The baseline and HSWL-A runs use the same training budget, seed, model, tokenizer and optimizer settings; only Semantic-ID loss weights differ. Position accuracy uses a fixed 4,096-example training subset, while end-to-end H@K uses the same fixed 1,024-example evaluation subset for both models.

| Model | RQ1 Acc | RQ2 Acc | RQ3 Acc | DEDUP Acc | H@5 | H@10 |
|---|---:|---:|---:|---:|---:|---:|
| Equal-weight baseline | 3.760% | 10.327% | 13.159% | 92.725% | 0.001953 | 0.002930 |
| HSWL-A | 3.760% | **13.257%** | **14.282%** | 92.749% | 0.001953 | 0.002930 |

HSWL-A improves RQ2/RQ3 teacher-forced token accuracy at 5k steps, but this improvement does **not** translate into an observed end-to-end HitRate gain on the 1,024-example evaluation subset. The repository reports this as a negative end-to-end result rather than claiming a recommendation-quality improvement.

## Installation

```bash
pip install -r requirements.txt
```

For the Amazon Beauty decoder experiments, download the upstream pretrained RQ-VAE and place it at:

```text
trained_models/rqvae_amazon_beauty/checkpoint_high_entropy.pt
```

Pretrained checkpoint: https://huggingface.co/edobotta/rqvae-amazon-beauty

On Apple Silicon, commands that encounter unsupported MPS operators can use:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 <command>
```

## Reproducing the main experiments

### Equal-weight baseline (5k)

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 MPS_LAUNCH_BLOCKING=1 \
python train_decoder.py configs/decoder_amazon_baseline_5k_seed2026.gin
```

### HSWL-A (5k)

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

Use `analyze_token_loss_mps.py` for evaluation-set position analysis.

## Implementation notes

- `model_jagged_mode=False` is used for the reported MPS decoder experiments because the original jagged/nested attention path is not fully supported on MPS.
- `ops/triton/jagged.py` retains the Triton path when available and provides a pure-PyTorch fallback otherwise.
- Generation retains stochastic candidate selection with `torch.multinomial`.
- HSWL defaults to equal weights, preserving compatibility with existing decoder checkpoints.
- Large datasets, experiment outputs and model checkpoints are intentionally excluded from version control.

## Repository layout

```text
configs/                     experiment configurations
modules/                     RQ-VAE, Transformer and retrieval model
ops/triton/jagged.py         Triton path + PyTorch fallback
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

- Experiments use a resource-constrained Apple Silicon setup rather than the original paper's accelerator setup or full training budget.
- Decoder experiments use the upstream pretrained RQ-VAE rather than a locally trained full RQ-VAE.
- The 5k HSWL comparison uses one controlled seed and is not a multi-seed statistical result.
- The 1,024-example H@K comparison is a screening subset. Full 22,363-example results above are reported for the 10k/20k/30k baseline checkpoints.
- HSWL improves intermediate token prediction in the reported experiment but does not establish an end-to-end recommendation improvement.

## Upstream and references

This repository is derived from Edoardo Botta's MIT-licensed RQ-VAE generative-retrieval implementation. The original copyright notice is preserved in `LICENSE`.

- Upstream implementation: https://github.com/EdoardoBotta/RQ-VAE
- Rajput et al., *Recommender Systems with Generative Retrieval*: https://arxiv.org/abs/2305.05065
- Jang et al., *Categorical Reparameterization with Gumbel-Softmax*: https://openreview.net/forum?id=rkE3y85ee
- Fifty et al., *Restructuring Vector Quantization with the Rotation Trick*: https://arxiv.org/abs/2410.06424
