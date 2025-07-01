# Tokenizer Leakage in LLMs: A Systematic Investigation

This repository contains the code, experiments, and analysis for a research project investigating **data leakage from tokenizers trained on validation data**, and its **effect on downstream validation performance in language models**.
This is a solo project that I started out of curiosity and to gain invaluable experience into training LLMs and engineering behind it. This also serves as a way to learn how to conduct rigorous experimentation with scientific integrity and honesty.

> ⚠️ This project is ongoing. Check the [dev log](dev_log.md) for the latest updates and experiments.

---

##  Motivation

When training language models, it is common to use tokenizers trained on the same data used to construct training and validation splits. This introduces a potential source of **leakage**: the tokenizer may encode knowledge of the validation set structure, biasing evaluation metrics.

This project asks:

- **Does training a tokenizer on the validation data lead to measurable performance gains on the validation set?**
- **Is this effect amplified by model size?**
- **Can such a leakage be detected through differences in token distributions, length, or compression ratios?**

I aim to quantify and explain this effect through controlled experiments.

---

## Author 

Fadi Benzaima 
Computer Science Student at ESI-SBA. 

