# PatchCon

**PatchCon: Patch-aware Weighted Supervised Contrastive Learning for Vulnerability Detection**

Chaomeng Lu, Pooya Rostami Mazrae, Bert Lagaisse (DistriNet, KU Leuven)

PatchCon is a CodeBERT-based approach for function-level vulnerability detection. It uses vulnerability-fixing patches as supervision and combines single-function classification with patch-aware supervised contrastive learning. Vulnerable–patched function pairs are treated as hard negatives with a larger weight than ordinary negatives, which helps the model capture the subtle security-relevant changes introduced by fixes.

This repository contains the implementation, the data and splits, the training/testing logs, the per-sample predictions of PatchCon and its variants, and the scripts used for the analyses in the paper. The trained model is too large for GitHub and is available on [figshare](https://figshare.com/s/cb1d02d2cea2f6582744) (see [Trained Model](#trained-model)); the figshare record also contains the complete package.

## Approach

PatchCon trains on paired code samples with two objectives:

1. **Classification loss** for predicting whether a function is vulnerable or safe.
2. **Patch-aware supervised contrastive loss** for shaping the embedding space using pair relationships.

The contrastive component uses the following pair types:

| Pair type (paper) | Name in the code | Description | Role |
|---|---|---|---|
| `same_safe` | `same_safe` | two safe functions | positive pair |
| `same_vuln` | `same_vuln` | two vulnerable functions | positive pair |
| `random_neg` | `random_neg` | random safe/vulnerable pair | ordinary negative pair |
| `patch` | `repair` | vulnerable function and its patched version | hard negative pair (weight $w_p$) |

![PatchCon_architecture](<PatchCon_Architecture.png>)

## Main Results

Overall performance on the PairVul-Linux test set (Table 1 of the paper). PACC (Pairwise Accuracy) is the proportion of vulnerable–patched pairs in which the vulnerable function is predicted as vulnerable and its patched version as safe; MCG (Mean Confidence Gap) is the average difference between the predicted vulnerability probabilities of the vulnerable function and its patched version. Pair: uses vulnerable–patched function pairs (for Vul-RAG, to build its knowledge base); PW: patch-pair weighting; InfoNCE, Triplet, NT-Xent, and SupCon also denote the contrastive loss types.

| Approach | Backbone | Contrastive type | Pair | PW | ACC | Precision | MCC | AUPRC | PACC | MCG |
|---|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| UniXcoder | UniXcoder | × | × | × | 0.5023 | 0.5012 | 0.0482 | 0.5355 | 0.0046 | 0.0024 |
| GraphCodeBERT | GraphCodeBERT | × | × | × | 0.5069 | 0.5035 | 0.0486 | 0.5363 | 0.0185 | 0.0076 |
| CodeBERT | CodeBERT | × | × | × | 0.5174 | 0.5096 | 0.0596 | 0.5329 | 0.0486 | 0.0035 |
| ContraBERT | CodeBERT | InfoNCE | × | × | 0.5150 | 0.5096 | 0.0364 | 0.5312 | 0.0671 | 0.0026 |
| ReVeal<sup>a</sup> | GNN | Triplet | × | × | 0.5024 | 0.5017 | 0.0052 | 0.5022 | 0.0481 | 0.0000 |
| SCL-CVD | CodeBERT | SupCon | × | × | 0.5544 | 0.5540 | 0.1088 | 0.5561 | 0.1505 | 0.0271 |
| DELTA | CodeBERT | SimCLR NT-Xent | ✓ | × | 0.5729 | 0.5609 | 0.1487 | 0.5786 | 0.2060 | 0.0855 |
| Vul-RAG<sup>b</sup> | GPT-4o | × | ✓ | × | 0.5974 | 0.6004 | 0.1950 | – | 0.2923 | – |
| **PatchCon** | CodeBERT | Weighted SupCon | ✓ | ✓ | **0.6030** | **0.6018** | **0.2060** | **0.6239** | **0.3264** | **0.1838** |

<sup>a</sup> Graph construction failed for 16 of the 432 test pairs, so ReVeal is evaluated on the remaining 416 pairs.

<sup>b</sup> Computed from the per-sample outputs released by the authors of Vul-RAG for 585 of the 586 pairs in its own test split, with undetermined answers counted as incorrect; AUPRC and MCG cannot be computed from its binary outputs.

The PatchCon row can be recomputed with `python analysis/metrics.py results/pairvul_linux/patchcon.csv`. The baselines were run with the code released by their authors (see [Baselines](#baselines)).

## Directory Structure

```text
PatchCon/
├── data/
│   ├── linux/                 # PairVul-Linux: linux_train.csv, linux_val.csv, linux_test.csv
│   └── ventivul/              # VentiVul (Linux kernel subset): ventivul_linux_pairs.csv
├── patchcon/
│   ├── logs/                  # training and testing log of the full model
│   ├── saved_models/          # put the trained checkpoint here (download from figshare)
│   ├── test_results/          # per-sample test output written by main.py
│   ├── main.py                # training and testing entry point
│   ├── model.py               # CodeBERT-based model
│   ├── trainer.py             # training, evaluation, and metrics
│   ├── data_loader.py         # dataset and pair construction
│   ├── paircollator.py        # tokenization and chunk-based batching
│   ├── supcon_loss.py         # patch-aware supervised contrastive loss
│   ├── utils.py               # utility functions
│   └── config.yaml            # configuration of the reported model
├── results/
│   ├── pairvul_linux/         # per-sample predictions of PatchCon and its variants on the PairVul-Linux test set
│   │   └── logs/              # training/testing logs of the variants
│   ├── ventivul/              # per-sample predictions of PatchCon on VentiVul
│   └── MANIFEST.csv           # configuration, paper table, and metrics of every prediction file
└── analysis/                  # scripts that compute the metrics and analyses reported in the paper
```

## Data

### PairVul-Linux

`data/linux/` contains PairVul-Linux, derived from the PairVul benchmark released with Vul-RAG ([repository](https://github.com/knowledgerag4llmvuld/knowledgerag4llmvuld), MIT license). It has 8,628 functions (4,314 vulnerable–patched pairs) from 2,073 Linux kernel CVEs. We split it randomly at the pair level with an 8:1:1 ratio, so that both functions of a pair are always in the same partition:

| File | Functions | Pairs | CVEs |
|---|---|---|---|
| `linux_train.csv` | 6,902 | 3,451 | 1,772 |
| `linux_val.csv` | 862 | 431 | 367 |
| `linux_test.csv` | 864 | 432 | 370 |

Columns: `cve_id`, `pair_id` (shared by the vulnerable and the patched function of a pair), `Function` (source code), `Vulnerable` (1 = vulnerable, 0 = patched), `CWE_type`.

As in the original PairVul benchmark, the split is not grouped by CVE. `analysis/cve_overlap.py` reports the results on test pairs whose CVE is seen or unseen during training.

### VentiVul (Linux kernel subset)

`data/ventivul/ventivul_linux_pairs.csv` contains the Linux kernel subset of VentiVul, a temporally separated test set introduced in our previous work:

> C. Lu and B. Lagaisse. From Lab to Reality: A Practical Evaluation of Deep Learning Models and LLMs for Vulnerability Detection. Journal of Systems and Software, 2026 (in press). Preprint: [arXiv:2512.10485](https://arxiv.org/abs/2512.10485).

It contains 116 vulnerable–patched pairs from 100 Linux kernel CVEs published between May 1 and May 29, 2025 (20 CWE types), with the same columns as PairVul-Linux plus `link` (fixing commit) and `in_paper_subset`. Pairs whose two versions are identical after whitespace normalization, or whose functions also appear in PairVul-Linux, are excluded from the evaluation (pairs 39, 72, 78, 88, and 116); the remaining 111 pairs from 96 CVEs (`in_paper_subset = True`) are used in the paper. `analysis/ventivul_subset.py` recomputes this flag. VentiVul is used only for testing.

## Trained Model

The checkpoint of the full PatchCon model (`best_model.pt`) exceeds GitHub's file size limit. Download it from [figshare](https://figshare.com/s/cb1d02d2cea2f6582744) and place it at `patchcon/saved_models/best_model.pt`, the path set by `checkpoint_path` in `patchcon/config.yaml`.

## Requirements

```bash
conda create -n patchcon python=3.12.4 -y
conda activate patchcon
pip install torch==2.3.1 transformers==4.37.2 pandas numpy scikit-learn tqdm pyyaml
```

The reported experiments use PyTorch 2.3.1, Transformers 4.37.2, Python 3.12.4, and CUDA 11.8 on NVIDIA H100 GPUs. Install the PyTorch build that matches your CUDA version.

## Configuration

`patchcon/config.yaml` contains the configuration of the reported model. Relative paths are resolved with respect to `patchcon/`.

| Setting | Value | Where |
|---|---|---|
| encoder | `microsoft/codebert-base` | `main.py` |
| learning rate / batch size | 2e-5 / 16 (AdamW) | `config.yaml` |
| epochs | at most 10, early stopping on validation AUPRC (patience 3) | `config.yaml` |
| SupCon temperature $\tau$ | 0.07 | `config.yaml` |
| SupCon loss weight $\lambda$ | 0.01 (`alpha_supcon`) | `config.yaml` |
| hard-negative weight $w_p$ | 2 (`"repair"` in `SUPCON_PAIR_WEIGHTS`) | `supcon_loss.py` |
| margin $m$ of the negative penalty | 0 (`hard_neg_margin`) | `supcon_loss.py` |
| projection dimension $k$ | 128 | `model.py` |
| classification head | LayerNorm, Dropout 0.1, Linear(768, 2) on the max-pooled function representation | `model.py` |
| chunking / MP size | at most 4 chunks of 512 tokens (510 code tokens each, stride 480), covering up to 1,950 code tokens; "MP size 2048" in the paper denotes 4 x 512 | `paircollator.py` (training), `data_loader.py` (testing) |
| pairs per type (train / validation) | at most 5,000 / 500 | `config.yaml` |
| pair balancing | `balance_cls_pairs: false`, `balance_all_pairs: false` | `config.yaml` |
| pair-classification loss | disabled (`use_cls: false`); an optional auxiliary loss that is not used in the paper | `config.yaml` |
| random seed | 42 | `config.yaml` |

`config.yaml` overrides two code defaults: `balance_cls_pairs` (default `true`) and `use_cls` (default `true` in `trainer.py`). The reported model was trained with both set to `false` (see the `[Pair Summary]` at the beginning of `patchcon/logs/train_test.log`); keep these settings to reproduce the paper.

## Reproducing the Experiments

Run all commands from the implementation directory:

```bash
cd PatchCon/patchcon
```

**Train from scratch** (the script also tests the best checkpoint at the end):

```bash
python main.py --mode train --config config.yaml 2>&1 | tee logs/train.log
```

**Test the provided model**: download the checkpoint (see [Trained Model](#trained-model)) to `saved_models/best_model.pt`, then run

```bash
python main.py --mode test --config config.yaml 2>&1 | tee logs/retest.log
```

The per-sample test output is written to `test_results/` and can be compared with `results/pairvul_linux/patchcon.csv`. To test on VentiVul, set `test_csv` to `../data/ventivul/ventivul_linux_pairs.csv`.

**Variants** (Tables 3–7 of the paper) are obtained by changing `use_supcon`/`use_single`, `alpha_supcon`, the `repair` weight in `supcon_loss.py`, the maximum number of chunks (`max_chunks` in `paircollator.py` for training and in `SingleFunctionDataset` in `data_loader.py` for testing; 1, 2, 3, and 4 chunks correspond to MP sizes 512, 1024, 1536, and 2048), the pair types used in `data_loader.py`, or the encoder in `main.py`; `results/MANIFEST.csv` lists the configuration of every run.

Retraining on other GPUs or software versions may give slightly different numbers because of nondeterministic GPU operations; the provided checkpoint reproduces the reported predictions.

## Per-sample Results

Each prediction file has one row per function: `id` (pair id), `true_label`, `pred_label`, `sample_type`, `prob_0`, `prob_1`. The rows of the PairVul-Linux files follow the order of `data/linux/linux_test.csv`.

- `results/pairvul_linux/`: 20 runs of PatchCon and its variants (ablation, backbones, hyperparameter sensitivity). Runs that appear in several tables are stored once; `results/MANIFEST.csv` maps every file to its configuration and to the tables and sections of the paper, together with its metrics.
- `results/ventivul/`: predictions of the full PatchCon model on VentiVul, on the 111 evaluated pairs (Table 9) and on all 116 pairs.

AUPRC is computed from `prob_1`. The AUPRC values in the training/testing logs are computed from the predicted labels and therefore differ from the values in the paper and in `MANIFEST.csv`.

## Analysis Scripts

Run from the package root (`PatchCon/`); the scripts only need `pandas`, `numpy`, and `scikit-learn` (and `transformers` for `--single-chunk`).

| Script | Paper | Example |
|---|---|---|
| `analysis/metrics.py` | Tables 1–7 and 9: ACC, Precision, Recall, F1, MCC, AUPRC, PACC, MCG, P-V/P-B/P-R | `python analysis/metrics.py results/pairvul_linux/patchcon.csv` |
| `analysis/bootstrap_ci.py` | Table 8: 95% bootstrap CIs (1,000 repetitions, seed 42); `--baseline` for difference CIs | `python analysis/bootstrap_ci.py results/pairvul_linux/patchcon.csv` |
| `analysis/cve_overlap.py` | Table 10: PACC and MCG on all, seen-CVE, and unseen-CVE test pairs | `python analysis/cve_overlap.py results/pairvul_linux/patchcon.csv` |
| `analysis/fine_grained.py` | RQ4 and RQ2: PACC by CWE type, function length, patch size; single-chunk pairs | `python analysis/fine_grained.py results/pairvul_linux/patchcon.csv --single-chunk` |
| `analysis/ventivul_subset.py` | Datasets: VentiVul pairs used in the evaluation | `python analysis/ventivul_subset.py` |

The scripts accept any prediction file in the same format, so predictions of other models on `data/linux/linux_test.csv` can be evaluated in the same way.

## Baselines

The baselines in the paper were run with the code released by their authors; their results are not included in this package.

| Baseline | Source |
|---|---|
| CodeBERT, GraphCodeBERT, UniXcoder | https://github.com/microsoft/CodeBERT |
| ContraBERT | https://github.com/shangqing-liu/ContraBERT |
| ReVeal | https://github.com/VulDetProject/ReVeal |
| SCL-CVD | https://github.com/AI4CVD/SCL4CVD |
| DELTA | https://github.com/XUPT-SSS/DELTA |
| Vul-RAG | https://github.com/knowledgerag4llmvuld/knowledgerag4llmvuld (per-sample outputs of the GPT-4o version) |

## License

The code and the results in this package are released under the MIT License (see `LICENSE`). PairVul-Linux is derived from PairVul, released with Vul-RAG under the MIT License (Copyright (c) 2024 KnowledgeRAG4LLMVulD).

## Citation

```bibtex
@misc{lu2026patchcon,
  title   = {PatchCon: Patch-aware Weighted Supervised Contrastive Learning for Vulnerability Detection},
  author  = {Lu, Chaomeng and Rostami Mazrae, Pooya and Lagaisse, Bert},
  year    = {2026}
  eprint={2611.06583},
  archivePrefix={arXiv},
  primaryClass={cs.SE},
  url={https://arxiv.org/abs/2611.06583}, 
}
```
