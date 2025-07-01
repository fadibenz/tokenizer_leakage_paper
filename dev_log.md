# 2025-07-01
- Set up the initial repository, used `uv` for dependency handling.
- Added DVC to version control data, I used a local remote file (no cloud credits)
- Implemented data loading utilities from scratch with memory-mapping and batch random sampling for training from corpus, data is `uint16` for efficiency.
- Added seeding utils for reproducibility and modular functions for creating models and configuration files.
- Implemented a cosine annealing scheduler with warmup from scratch and tested it using CS336 public tests. 
- Implemented evaluation script that goes, deterministically, over dataset and calculates loss and perplexity
