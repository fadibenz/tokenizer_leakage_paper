# 2025-07-01
- Set up the initial repository, used `uv` for dependency handling.
- Added DVC to version control data, I used a local remote file (no cloud credits)
- Implemented data loading utilities from scratch with memory-mapping and batch random sampling for training from corpus, data is `uint16` for efficiency.
- Added seeding utils for reproducibility and modular functions for creating models and configuration files.
- Implemented a cosine annealing scheduler with warmup from scratch and tested it using CS336 public tests. 
- Implemented evaluation script that goes, deterministically, over dataset and calculates loss and perplexity.
- Implemented training loop with mixed-precision training (using torch context manager) and wandb logging.

# 2025-07-02
- Finished script of the full experiment training.  
- Finished training of the two tokenizers (clean and leaky) using my own implementation of BPE tokenizer (tested using CS336 test suite)
- BPE training took 2h for the clean version and around 3h for the leaky one (with less than 16GB RAM). 
- Tokenized the training text using both tokenizers (throughput of around 0.19MB/s, took 6 hours each)
- Did some napkin math to estimate running time:
  - Considering a batch size of $32$ with context length of $1024$ and a decent $50k$ steps for training, we get $`\text{num\_tokens} = 1.6B \text{tokens}`$
  - Using the scaling relationship for training $\text{Total FLOPs} \approx 6 \times (\text{tokens}) \times (\text{parameters})$, we get: 
    - For $30M$: $\approx \mathbf{288{,}000}$ TFLOPs
    - For $60M$: $\approx \mathbf{576{,}000}$ TFLOPs
    - For $125M$: $\approx \mathbf{1{,}200{,}000}$ TFLOPs
    - For $250M$: $\approx \mathbf{2{,}400{,}000}$ TFLOPs
  - Training options include kaggle free options: Tesla P100, Tesla T4 $\times$ 2 and TPU v3-8, some simple calculations Assuming MFU of $0.4$ (P100), $0.45$ (T4×2), $0.6$ (TPU v3-8):
  
      | Model  | P100       | T4 ×2     | TPU v3-8  | 
      |--------|------------|-----------|-----------| 
      | $30M$  | ~22.2 hrs  | ~11.0 hrs | ~1.3 min  | 
      | $60M$  | ~44.4 hrs  | ~22.0 hrs | ~2.7 hrs  | 
      | $125M$ | ~92.6 hrs  | ~46.0 hrs | ~5.6 hrs  | 
      | $250M$ | ~185.2 hrs | ~92.0 hrs | ~11.1 hrs |
  - TPU v3-8 has a total memory of $128GB$, training the model on FP32 with AdamW, we can fit up to $\text{Max parameters} = \frac{128 \times 10^9}{16} = 8 \times 10^{9} \text{ (8B parameters)}$, so assuming full utilization, we can train up to 250M parameters comfortably.
- I will stick with TPU v3-8 for training, since it offers great training time, I will need to change the code to work on TPU.
- **NOTE:** Apparently Kaggle does not give you eight cores (some say 4 or even 1), so I might need to scale down my experiments.


# 2025-07-03
- Did some literature review for hyperparameters and architecture choices. 
- Wrote configs for different model sizes, using this formula: 
  - Non-Embedding params: $`\text{vocab\_size} \times \text{d\_model}`$ 
  - Embedding params: $`4 \times \text{d\_model}^2 \ \text{(attention)} + 3 \times \text{d\_model} \times \text{d\_ff} \ \text{(FFN)} + \text{d\_model} \ \text{(RMSNorm)}`$ 
  - Tried to keep these ratios: 
    - $`\text{d\_ff} \approx \frac{8}{3} \times \text{d\_model}`$
    - $`\frac{\text{d\_model}}{\text{n\_layers}} \approx 50 - 100`$
    - $`\text{num\_heads} = \frac{\text{d\_model}}{64}`$
- Migrated code to TPU, my OpenMPI knowledge came in handy, key changes include:
  - **`DistributedSampler`**: Replaced manual data loading with `DistributedSampler` for proper and efficient data distribution across TPU cores.
  - **`MpDeviceLoader`**: Integrated `MpDeviceLoader` to handle asynchronous data transfer to TPU devices.
  - **`torch_xla.core.xla_model` (xm) API**:
    - Utilized `xm.xla_device()` for correct device placement. 
    - Implemented `xm.is_master_ordinal()` to ensure single-instance operations (e.g., logging, checkpointing, `tqdm` updates) from the master core. 
    - Employed `xm.all_reduce()` in `evaluate_perplexity` for accurate aggregation of metrics across all TPU cores. 
    - Switched to `xm.optimizer_step()` for efficient compilation and execution of the training step on TPUs. 
    - Added `xm.rendezvous()` for synchronization at critical points in the training loop.
  - **`spawn` (xmp)**: Adapted the main entry point to use `xmp.spawn` for launching the distributed training process across multiple TPU cores.
  - **`torch.autocast(dtype=torch.bfloat16)`**: Enabled `bfloat16` mixed-precision training for improved performance and memory efficiency on TPUs.c
- Wrote test suits for model creation, made sure that the shapes match and the model is randomly initialized with the provided configuration file.

# 2025-07-04

- Started full integration test on TPU to fix potential bugs.
- Fixed several bugs:
  - wrong typing in command line-arguments.
  - improper use of `xm.mark_step()`.
  - wrong type casting in data loading.
- Learned how to work with TPU in kaggle:
  - DON'T create a virtual environment as it leads to dependency problems, install directly since we already have a dedicated environment.
  - when using `xm.spawn` it needs to be the first executed function.
  - Kaggle has a problem in which you need to run `os.environ.pop('TPU_PROCESS_ADDRESSES')` before using TPU 
- Added `stride` in data loading to allow for granular choice on overlapping window size in validation vs. training.
- Read about different validation strategies and compromises.
- Finished tokenization of all files including validation and test.


# 2025-07-05
- Dealt with memory problems by adding explicit memory management and garbage collection. 
- Decided on using a non-overlapping window for validation during training and a window of context-length - 128 for final validation and test scores.
- Tried profiling with torch.profiler but didn't work as expected so decided to use torch_xla.debug
- Added syncfree optimizer and torch_xla.amp.autocast instead of torch.autocast. 
- Fixed logging and pbar display.

# 2025-07-06
- Went into the pytorch-XLA rabbit hole, very hard to work with, cryptic logs, sudden crashes and more.
- I spent the whole day trying to debug and the kaggle overhead added more friction (I needed to push each change to see results)
  - You need explicit try-catch blocks to see errors or most things silently fail.
  - using spawn with `fork` causes problems and race conditions, you need to use `spawn`.
  - using `num_workers > 0` and `presistent_workers = True` causes training to fail silently.
- Tried to optimize evaluation by reducing number of calls `mark_step()`and garbage collection, which lead to faster evaluation. 
- Training still silently fails sometimes, I'm still investigating why. more like trash-xla.


# 2025 - 07 - 08
- Fixed evaluation deadlock by adding an explicit barrier before reducing and used mesh_reduce instead of all_reduce.
- Fixed training by only calling `.item()` after explicitly calling `mark_step()`
- Removed try-catch and unnecessary branching conditions to speed up training.
- Apparently, calling `mark_step()` explicitly in each iteration significantly optimizes training speed. 
- Spent an outrageous amount of time trying to figure out why my evaluation keeps failing, fixed it by calling `mark_step()` once after evaluation finishes 
- Fixed inconsistencies in my config files and reduced learning rate for stable learning (loss showed spikes)