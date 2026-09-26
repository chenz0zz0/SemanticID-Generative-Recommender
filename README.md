# SemanticID Generative Recommender

A generative recommendation system based on **Semantic IDs**, using **RQ-VAE** for item tokenization and a **Transformer** for autoregressive next-item generation.

This repository extends an existing PyTorch implementation of *Recommender Systems with Generative Retrieval* with hierarchical Semantic-ID loss weighting, reproducible experiments, evaluation utilities, and compatibility improvements.

## Overview

Generative retrieval formulates recommendation as a sequence generation problem. Instead of directly scoring every candidate item, each item is represented by a short sequence of discrete Semantic IDs, and the recommendation model generates the Semantic ID of the next item.

The system contains two stages:

1. **Semantic-ID tokenization** — an RQ-VAE converts item representations into hierarchical discrete codes.
2. **Generative retrieval** — a Transformer models users' historical Semantic-ID sequences and autoregressively generates the Semantic ID of the next item.

```text
Item representations
        |
        v
      RQ-VAE
        |
        v
Hierarchical Semantic IDs
        |
        v
User interaction sequence
        |
        v
    Transformer
        |
        v
Next-item Semantic ID
        |
        v
Recommended item
```

For the Amazon Beauty setup used in the experiments, an item is represented by four components:

```text
[RQ1, RQ2, RQ3, DEDUP]
```

`RQ1`–`RQ3` are produced by the residual quantization layers. `DEDUP` is an additional identifier used to distinguish items that would otherwise share the same quantized representation.

## Hierarchical Semantic-ID Weighted Loss

The baseline decoder applies equal weight to every Semantic-ID position:

```text
[1.0, 1.0, 1.0, 1.0]
```

Because different levels of a hierarchical Semantic ID can have different prediction difficulty, this project implements **Hierarchical Semantic-ID Weighted Loss (HSWL)**.

The HSWL-A configuration used in the controlled experiment is:

```text
[1.5, 1.25, 0.75, 0.5]
```

The four weights sum to the same value as the baseline weights, limiting changes in the overall loss scale while shifting more optimization emphasis toward earlier semantic levels.

The training objective is:

```text
L = mean(
      w1 * CE(RQ1)
    + w2 * CE(RQ2)
    + w3 * CE(RQ3)
    + w4 * CE(DEDUP)
)
```

Unweighted position-wise losses are retained separately for analysis.

## Experiments

Experiments were conducted on the **Amazon Reviews Beauty** dataset. Decoder experiments use the released pretrained Amazon Beauty RQ-VAE checkpoint to generate Semantic IDs.

The primary recommendation metrics are full Semantic-ID **HitRate@K (H@K)**. Position-wise cross-entropy and accuracy are additionally used to analyze the behavior of individual Semantic-ID levels.

### Decoder Training Budget

The baseline decoder was evaluated at 10k, 20k, and 30k training iterations.

| Decoder checkpoint | H@1 | H@5 | H@10 |
| --- | ---: | ---: | ---: |
| 10k | 0.002325 | 0.008943 | 0.016858 |
| 20k | 0.002102 | 0.007065 | 0.016500 |
| 30k | 0.002325 | 0.007557 | 0.016500 |

Although training loss continued to decrease with additional optimization, the evaluated full-ID HitRate did not improve correspondingly. This illustrates that lower teacher-forced training loss does not necessarily translate into better end-to-end autoregressive retrieval.

### HSWL Controlled Experiment

A controlled 5k-iteration comparison was performed using the same random seed and training setup.

| Model | RQ1 Acc. | RQ2 Acc. | RQ3 Acc. | H@5 (1024) | H@10 (1024) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 3.760% | 10.327% | 13.159% | 0.001953 | 0.002930 |
| HSWL-A | 3.760% | 13.257% | 14.282% | 0.001953 | 0.002930 |

HSWL-A improved teacher-forced prediction accuracy at the intermediate RQ2 and RQ3 levels under this training budget. However, the improvement did **not** translate into higher end-to-end HitRate on the evaluated 1024-sample subset.

This result is retained as a negative result rather than being presented as an end-to-end recommendation improvement.

## Key Findings

The experiments highlight several practical observations:

- Semantic-ID levels exhibit different prediction difficulty.
- Reweighting the hierarchical loss changes how optimization is distributed across Semantic-ID positions.
- HSWL-A improved RQ2 and RQ3 token-level accuracy in the controlled 5k experiment.
- Better token-level teacher-forced metrics did not automatically produce better autoregressive recommendation HitRate.
- Extending baseline decoder training from 10k to 30k iterations reduced training loss but did not improve the evaluated full-ID HitRate.

These results emphasize the gap between token-level optimization and end-to-end generative retrieval quality.

## Project Structure

```text
.
├── configs/                         # Gin experiment configurations
├── data/                            # Dataset processing code
├── distributions/                   # Distribution utilities
├── modules/                         # RQ-VAE and Transformer modules
├── ops/                             # Low-level operators and fallbacks
├── train_rqvae.py                   # RQ-VAE training
├── train_decoder.py                 # Generative retrieval training
├── eval_decoder_subset_mps.py       # Subset HitRate evaluation
├── analyze_token_loss_mps.py        # Position-wise evaluation analysis
├── analyze_train_token_loss_mps.py  # Train-subset token analysis
├── test_generation_mps.py           # Generation smoke test
├── requirements.txt
└── LICENSE
```

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

The project uses `gin-config` for experiment configuration.

## Training

### RQ-VAE

Train the RQ-VAE tokenizer with:

```bash
python train_rqvae.py configs/rqvae_amazon.gin
```

A lightweight smoke-test configuration is also provided:

```bash
python train_rqvae.py configs/rqvae_amazon_smoke.gin
```

### Decoder Baseline

A baseline decoder can be trained with:

```bash
python train_decoder.py configs/decoder_amazon_baseline.gin
```

The repository also contains configurations used for different decoder training budgets.

### HSWL

The controlled HSWL-A configuration can be run with:

```bash
python train_decoder.py configs/decoder_amazon_hswl_a_5k_seed2026.gin
```

The corresponding baseline comparison is:

```bash
python train_decoder.py configs/decoder_amazon_baseline_5k_seed2026.gin
```

Both configurations use the same seed and training budget so that the loss-weighting strategy is the primary experimental variable.

## Evaluation

Subset generation evaluation can be run with:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python eval_decoder_subset_mps.py \
  --checkpoint <decoder-checkpoint> \
  --num-samples 1024 \
  --batch-size 2
```

Position-wise Semantic-ID metrics can be analyzed with the provided analysis scripts.

Checkpoint paths are intentionally not bundled with the repository. Download or train the required models locally and configure their paths before running the corresponding experiment.

## Implementation Notes

The project includes several engineering changes used during the experiments:

- configurable Semantic-ID loss weights;
- explicit Python and PyTorch random seeds;
- corrected resumed-training iteration handling;
- corrected final-checkpoint saving;
- a pure PyTorch fallback for Triton-dependent jagged operations;
- a padded Transformer execution path for environments where jagged/nested attention is not fully supported.

The project was tested with PyTorch's **MPS** backend. Unsupported operations can use CPU fallback by setting:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Limitations

- The experiments use the released pretrained Amazon Beauty RQ-VAE rather than reproducing the full tokenizer training budget from scratch.
- Decoder training budgets are substantially smaller than large-scale paper training settings.
- The reported HSWL experiment demonstrates token-level changes but does not establish an improvement in end-to-end recommendation HitRate.
- The 1024-sample HSWL evaluation is a subset evaluation and should not be interpreted as a full benchmark result.
- Results should therefore be interpreted as reproduction and controlled experimental observations rather than state-of-the-art claims.

## Upstream and Attribution

This repository is derived from the open-source PyTorch RQ-VAE / generative retrieval implementation by **Edoardo Botta** and retains the original **MIT License** and copyright notice.

The project builds on the generative retrieval approach introduced in:

**Recommender Systems with Generative Retrieval**  
Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan H. Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Q. Tran, Jonah Samost, Maciej Kula, Ed H. Chi, Maheswaran Sathiamoorthy.

## References

- Rajput et al., *Recommender Systems with Generative Retrieval*, 2023.
- Jang, Gu, and Poole, *Categorical Reparameterization with Gumbel-Softmax*, 2017.
- Fifty et al., *Restructuring Vector Quantization with the Rotation Trick*, 2024.
- Original RQ-VAE / generative retrieval PyTorch implementation by Edoardo Botta.

## License

This project retains the upstream MIT License. See `LICENSE` for details.
