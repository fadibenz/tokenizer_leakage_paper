# Tokenizer Leakage in LLMs: A Systematic Investigation

**Blog & experiments:** https://fadibenz.vercel.app/blog/data-leakage-tokenization

This repository contains the code, experiments, and analysis for a research project investigating **data leakage from tokenizers trained on validation data**, and its **effect on downstream validation performance in language models**.
This is a solo project that I started out of curiosity and to gain invaluable experience into training LLMs and engineering behind it. 
This also serves as a way to learn how to conduct rigorous experimentation with scientific integrity and honesty.

This repository contains the implementation, experiment scripts,logs and analysis for a controlled study of tokenizer “leakage”: training BPE tokenizers on train + val vs train only, and measuring downstream effects on model validation/test perplexity across several LLaMA-style model sizes.


###  Motivation

When training language models, it is common to use tokenizers trained on the same data used to construct training and validation splits. This introduces a potential source of **leakage**: the tokenizer may encode knowledge of the validation set structure, biasing evaluation metrics.

This project asks:

- **Does training a tokenizer on the validation data lead to measurable performance gains on the validation set?**
- **Is this effect amplified by model size?**
- **Can such a leakage be detected through differences in token distributions, length, or compression ratios?**

I aim to quantify and explain this effect through controlled experiments.


### Highlights / TL;DR

+ **Goal:** quantify whether a tokenizer trained on validation data introduces a measurable evaluation bias (perplexity) and whether the effect scales with model size. 
+ **Datasets:** OpenWebText subset (train/val/test splits described in paper). 
+ **Architectures:** LLaMA-style decoder-only Transformer (30M, 60M, 125M configurations). 
+ **Tokenizers:** custom optimized BPE (32k vocab); two conditions — Clean (train only) and Leaky (train + val).

> Result summary: no practical bias detected on large, in-domain OpenWebText (ΔTest-PPL ≤ 0.02 in our runs; single-seed;).


### Repository layout

``` graphql
/
├─ tokenizer_leakage/           # Core code: model, training loop, evaluation, utils
│   ├─ configs/                 # All hyperparameter YAMLs (per model/run)
│   ├─ scripts/                 # Top-level scripts to training and evaluation 
│   ├─ notebooks/               # Notebooks for analysis & diagnostics
│   ├─ results/                 # Checkpoints, logs, plots (large files typically in LFS)
│   ├─ src/                     # data, training and evaluation code
│   └─ tokenizer/               # Link to tokenizer repo or local tokenizer artifacts
├─ tests/                       # Unit/integration tests for utils and model code
└─ README.md                    # <-- this file
```

### Reproducibility & hyperparameters

+ All exact hyperparameters (optimizer settings, LR schedules, warmup steps, batch sizes, seeds) live under configs/ and are included as YAMLs.
+ All runs log to Weights & Biases (W&B).

---

## Author 

Fadi Benzaima 
Computer Science Student at ESI-SBA. 

