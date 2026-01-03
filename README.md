# ML Training and Inference Suite

This repository contains a suite of tools and scripts for training and inference of machine learning models.
The suite includes functionalities for data preprocessing, model training, evaluation, and deployment.

## Why This Setup? (For Jupyter Users)

If you're used to training models in Jupyter notebooks, you might wonder why you need all this infrastructure. Here's the short version:

**Jupyter is great for:**
- Prototyping and experimentation
- Interactive debugging and visualization
- Quick iterations on small datasets

**This setup is designed for:**
- Training that takes hours/days (you can't keep a notebook open that long)
- Using multiple GPUs (distributed training)
- Running on HPC clusters with job schedulers (SLURM)
- Reproducible experiments with version-controlled configs
- Automatic recovery from crashes or time limits

**Key differences from Jupyter:**

| Jupyter Notebook | This Training Suite |
|-----------------|---------------------|
| Run cells manually | Run a single script that handles everything |
| Lose progress if kernel dies | Automatic checkpointing and resumption |
| Print statements for logging | Structured logging + WandB dashboards |
| Hardcoded hyperparameters | YAML config files (easy to track and modify) |
| Single GPU | Multi-GPU and multi-node support |
| Keep notebook open | Submit job and check results later |

The learning curve is worth it: once you understand this setup, you can train models for days on powerful hardware without babysitting them.

## Features
- Modular design for easy integration and extension
- torch.compile (with memory constraints)
- Automatic Mixed Precision (AMP) support
- Distributed training capabilities
- WandB integration for experiment tracking
- Configurable via YAML files
- Checkpointing and resuming training support
- Time keeping for graceful shutdowns and resuming on HPCs

## Key Concepts Explained

This section explains the advanced features you'll encounter. If you've only used basic PyTorch in Jupyter, these might be new to you.

### Automatic Mixed Precision (AMP)

**What it is:** By default, PyTorch uses 32-bit floating point numbers (float32) for all computations. AMP automatically uses lower precision (float16 or bfloat16) where it's safe, while keeping float32 where precision matters.

**Why use it:**
- **2x faster training** (or more) on modern GPUs
- **Lower memory usage** = larger batch sizes
- **Almost no accuracy loss** when done correctly

**How it works in this codebase:**
```python
# Without AMP (what you might write in Jupyter):
output = model(x)
loss = criterion(output, target)
loss.backward()

# With AMP (what this codebase does):
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    output = model(x)
    loss = criterion(output, target)
scaler.scale(loss).backward()  # GradScaler prevents underflow
```

**Config options:**
```yaml
amp: true              # Enable/disable AMP
precision: bfloat16    # Use bfloat16 (recommended for A100/H100) or float16
```

**When to disable:** If you see NaN losses or unstable training, try `amp: false` to rule out precision issues.

### torch.compile

**What it is:** A PyTorch 2.0+ feature that JIT-compiles your model for faster execution. Think of it as an optimizer that fuses operations and generates optimized GPU kernels.

**Why use it:**
- **10-30% speedup** on many models (sometimes more)
- **No code changes required** to your model

**The catch:**
- First batch is slow (compilation time)
- Not all models compile successfully
- Some dynamic operations aren't supported

**Config options:**
```yaml
compile: true    # Enable torch.compile
mem_budget: 1    # Memory budget (1 = full memory, <1 = gradient checkpointing)
```

**Gradient checkpointing via mem_budget:** When `mem_budget < 1`, the compiled model trades compute for memory by recomputing activations during the backward pass instead of storing them all. Set to 0.5 to use roughly half the memory at the cost of ~20-30% more compute.

### Distributed Training (DDP)

**What it is:** Training on multiple GPUs simultaneously, where each GPU processes different batches and gradients are synchronized.

**Why use it:**
- **Near-linear speedup** with more GPUs (4 GPUs ≈ 4x faster)
- **Larger effective batch sizes** without memory limits

**How it works:**
1. Your model is copied to each GPU
2. Each GPU gets different data (via DistributedSampler)
3. Each GPU computes gradients independently
4. Gradients are averaged across all GPUs (all-reduce)
5. All GPUs update with the same averaged gradients

**Running with multiple GPUs:** Instead of `python train.py`, use:
```bash
# 4 GPUs on one machine
torchrun --standalone --nproc_per_node=4 run_training.py --config_path config.yaml
```

**Important:** The `batch_size` in config is **per GPU**. With 4 GPUs and batch_size=64, your effective batch size is 256.

### Weights & Biases (WandB)

**What it is:** A cloud service for tracking ML experiments. Think of it as a supercharged TensorBoard with collaboration features.

**Why use it:**
- **Track metrics** across runs without scrolling through terminal output
- **Compare experiments** side-by-side with interactive charts
- **Log hyperparameters** automatically for reproducibility
- **Share results** with collaborators via web dashboard

**Setup:**
1. Create a free account at [wandb.ai](https://wandb.ai)
2. Run `wandb login` or add your API key to `.env`
3. Set project/entity in config

**What gets logged:** Loss curves, learning rate, validation metrics, system stats (GPU memory, utilization).

### Checkpointing and Resumption

**What it is:** Saving the complete training state (model weights, optimizer state, learning rate scheduler, epoch number) so you can continue training later.

**Why it matters:**
- **HPC time limits:** Clusters often limit jobs to 24-48 hours. Checkpointing lets you chain jobs.
- **Crash recovery:** If training crashes at hour 23, you don't lose everything.
- **Experimentation:** Train for a while, then try different fine-tuning strategies.

**What gets saved:**
- Model weights
- Optimizer state (momentum, Adam statistics)
- Learning rate scheduler state
- Number of samples/batches trained
- GradScaler state (for AMP)

**Config options:**
```yaml
checkpoint:
  checkpoint_name: latest  # Load "latest", "best", or epoch number (e.g., "5")
  restart: true            # true = continue training, false = only load weights (for fine-tuning)
```

### Time Keeping for HPC

**What it is:** The training loop monitors elapsed time and gracefully stops before hitting the job time limit, saving a checkpoint first.

**Why it matters:** On SLURM clusters, if your job hits the time limit, it gets killed immediately (SIGKILL). Any unsaved progress is lost. This suite estimates how long epochs take and stops early to save.

**Config:**
```yaml
time_limit: "24:00:00"  # Should be slightly less than SLURM --time
```

### YAML Configuration

**What it is:** A human-readable format for configuration files. Better than hardcoding hyperparameters in Python.

**Why use YAML instead of command-line args:**
- **Readable:** Easy to see all settings at once
- **Version controllable:** Track config changes in git
- **Reproducible:** Copy the config to reproduce an experiment exactly
- **Complex structures:** Nested configs for model, optimizer, scheduler

**Example structure:**
```yaml
dataset:
  name: my_dataset
  train_split: 0.8

model:
  type: transformer
  num_layers: 12

optimizer:
  name: AdamW
  learning_rate: 1e-4
```

### Environment Variables (.env file)

**What it is:** A file containing machine-specific paths and secrets that shouldn't be in version control.

**Why use it:**
- **Security:** API keys stay out of git
- **Portability:** Same code works on laptop and HPC with different paths
- **Flexibility:** Easy to switch between machines

**Example `.env`:**
```bash
WANDB_API_KEY=your_secret_key_here
BASE_DIR=/home/user/ml-training-suite      # Path to this repo
DATA_DIR=/scratch/datasets                  # Where your data lives
RESULTS_DIR=/scratch/results                # Where checkpoints go
```

## Structure

```
ML-Training-Suite/
├── .env                          # Your environment variables (create this)
├── pyproject.toml                # Python dependencies (managed by uv)
├── ml_suite/
│   ├── data/
│   │   ├── dataset.py            # Dataset class (customize this)
│   │   └── dataloader.py         # DataLoader factory with DDP support
│   ├── models/
│   │   ├── model_utils.py        # get_model() function (customize this)
│   │   ├── loss_fns.py           # Loss functions (MSE, MAE, RMSE, etc.)
│   │   └── unet.py               # Example model architecture
│   └── train/
│       ├── train_base.py         # Core Trainer class (don't modify unless needed)
│       ├── eval.py               # Evaluator class
│       ├── run_training.py       # CLI entry point (config loading, setup)
│       ├── train.yml             # Example config (copy to results dir)
│       ├── scripts/
│       │   └── train_riv.sh      # SLURM job script template
│       └── utils/
│           ├── checkpoint_utils.py  # Save/load checkpoints
│           ├── lr_scheduler.py      # Chained LR scheduler
│           ├── optimizer.py         # Optimizer factory
│           ├── time_keeper.py       # HPC time limit handling
│           ├── wandb_logger.py      # WandB integration
│           └── logger.py            # Console logging
└── tests/                        # Unit tests
```

**Understanding the separation:**

The code is split into three layers:

1. **Pure Python logic** (`train_base.py`, `eval.py`): Clean, testable classes that take Python objects as input. No file I/O, no config parsing. This is what you'd write in a well-structured Jupyter notebook.

2. **CLI glue code** (`run_training.py`): Reads configs, loads checkpoints, sets up distributed training, creates objects, and passes them to the Trainer. This is the "ugly but necessary" code that connects configs to the pure logic.

3. **Shell scripts** (`scripts/train_riv.sh`): SLURM job submission and environment setup. Different for each cluster.

**When adapting for your project, you'll mainly modify:**
- `ml_suite/data/dataset.py` - Your data loading logic
- `ml_suite/models/model_utils.py` - Your model architectures
- Config YAML files - Your hyperparameters


## Instructions

### Step 1: Initial Setup

**Fork or copy the repository:**
```bash
git clone https://github.com/your-username/ML-Training-Suite.git
cd ML-Training-Suite
```

**Install uv (Python package manager):**

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. It's like pip + venv but much faster.

```bash
# On Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

**Install dependencies:**
```bash
uv sync --extra dev
```

This creates a virtual environment in `.venv/` and installs everything, including PyTorch with CUDA support. The exact versions are locked in `uv.lock` for reproducibility.

### Step 2: Create Your .env File

Create a `.env` file in the repository root:

```bash
# .env
WANDB_API_KEY=your_api_key_here     # Get from wandb.ai/settings
BASE_DIR=/path/to/this/repo          # Absolute path to ML-Training-Suite
DATA_DIR=/path/to/your/datasets      # Where your training data lives
RESULTS_DIR=/path/to/results         # Where checkpoints and logs go
```

**Why separate directories?**
- `BASE_DIR`: The code. Should be the same across machines (or cloned separately).
- `DATA_DIR`: Your datasets. Might be on a fast NVMe drive or shared filesystem.
- `RESULTS_DIR`: Checkpoints and logs. Often on scratch space that doesn't count against quota.

### Step 3: Set Up WandB (Recommended)

1. Create a free account at [wandb.ai](https://wandb.ai)
2. Go to Settings → API Keys and copy your key
3. Add it to your `.env` file
4. Create a new project on WandB (or the suite will create one automatically)

### Step 4: Add Your Model

Edit `ml_suite/models/model_utils.py`:

```python
def get_model(config: dict) -> torch.nn.Module:
    """Factory function that returns a model based on config."""
    model_type = config.get("type", "transformer")

    if model_type == "my_custom_model":
        return MyCustomModel(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            # ... other params from config
        )
    elif model_type == "transformer":
        # existing code...
```

**Tip:** Keep your model as a standard `nn.Module`. The training loop handles AMP, DDP wrapping, and compilation automatically.

### Step 5: Add Your Dataset

Edit `ml_suite/data/dataset.py`:

```python
def get_dataset(config: dict, split: str = "train") -> Dataset:
    """
    Factory function that returns a dataset.

    Args:
        config: Dataset configuration from YAML
        split: "train" or "valid"

    Returns:
        A PyTorch Dataset that returns (input, target) tuples
    """
    # Your dataset loading logic here
    # Must return (input_tensor, target_tensor) per item
```

**Important:** Your dataset must return `(input, target)` tuples. The training loop expects this format.

### Step 6: Configure Your Experiment

1. **Create a results directory for your experiment:**
```bash
mkdir -p $RESULTS_DIR/my_experiment
```

2. **Copy and edit the config:**
```bash
cp ml_suite/train/train.yml $RESULTS_DIR/my_experiment/train.yml
```

3. **Edit the config file.** Key sections to modify:

```yaml
# Dataset - your data parameters
dataset:
  name: my_dataset
  train_split: 0.8

# Model - must match get_model()
model:
  type: my_custom_model
  input_dim: 128
  hidden_dim: 256
  criterion: MSE

# Training parameters
batch_size: 64            # Per GPU!
total_updates: 100000     # Total gradient updates
updates_per_epoch: 1000   # Updates between evaluations

# Optimizer
optimizer:
  name: AdamW
  learning_rate: 1e-4
  weight_decay: 0.01

# WandB
wandb:
  enabled: true
  project: my-project
  entity: my-username
```

### Step 7: Run Training

**Local machine with single GPU:**
```bash
source .venv/bin/activate
python ml_suite/train/run_training.py --config_path $RESULTS_DIR/my_experiment/train.yml
```

**Local machine with multiple GPUs:**
```bash
source .venv/bin/activate
torchrun --standalone --nproc_per_node=4 ml_suite/train/run_training.py --config_path $RESULTS_DIR/my_experiment/train.yml
```

**On SLURM cluster:**
1. Edit `ml_suite/train/scripts/train_riv.sh`:
   - Set `#SBATCH --account=your_account`
   - Set `#SBATCH --gres=gpu:a100:4` (or your GPU type)
   - Update `sim_name` to match your experiment folder name

2. Submit the job:
```bash
sbatch ml_suite/train/scripts/train_riv.sh
```

### Step 8: Monitor Training

**Check logs:**
```bash
tail -f $RESULTS_DIR/my_experiment/training.log
```

**View WandB dashboard:**
Go to wandb.ai/your-username/your-project

**Resume from checkpoint:**
Set in your config:
```yaml
checkpoint:
  checkpoint_name: latest  # or "best" or epoch number like "5"
  restart: true
```

## Notes

### Updates vs Epochs

This suite uses **gradient updates** (batches) instead of epochs to measure training progress:

```yaml
total_updates: 100000       # Stop after 100k gradient updates
updates_per_epoch: 1000     # Run validation every 1000 updates
```

**Why?** Epochs are confusing when dataset size or batch size changes:
- 10 epochs with batch_size=32 on 10k samples = 3,125 updates
- 10 epochs with batch_size=64 on 50k samples = 7,812 updates

Updates are consistent regardless of dataset/batch size. If you prefer epochs, calculate: `updates = epochs * (dataset_size / batch_size)`.

### Learning Rate Scheduling

The suite supports **chained schedulers** - multiple phases that run sequentially:

```yaml
lr_scheduler:
  first_stage:      # Warmup: gradually increase LR
    name: LinearLR
    start_factor: 0.001
    end_factor: 1.0
    num_updates: 5000

  second_stage:     # Main training: cosine decay
    name: CosineAnnealingLR
    num_updates: -1    # -1 = use remaining updates
    end_factor: 0.01   # End at 1% of peak LR

  third_stage:      # Cooldown: decay to zero
    name: LinearLR
    end_factor: 0
    num_updates: 10
```

This creates the classic "warmup → cosine decay → cooldown" schedule that works well for most models.

### Fine-tuning vs Resuming

The `restart` parameter controls what happens when loading a checkpoint:

```yaml
checkpoint:
  checkpoint_name: latest
  restart: true   # Continue exactly where you left off
```

- `restart: true`: Load model, optimizer, scheduler, and training state. Training continues seamlessly.
- `restart: false`: Load only model weights. Use fresh optimizer/scheduler. Good for fine-tuning with different hyperparameters.

## Suggestions for Beginners

1. **Start simple:** Disable `compile` and `amp` initially. Get your model training first, then enable optimizations.

2. **Use WandB from day one.** Even for quick experiments. You'll thank yourself when you need to compare runs.

3. **Start with small `total_updates`** (e.g., 1000) to verify everything works before long runs.

4. **Default LR schedule:** Linear warmup (5% of training) + cosine annealing works for most models.

5. **If training is unstable:** Try gradient clipping with `max_grad_norm: 1.0`.

6. **Memory issues?** Reduce `batch_size` or set `mem_budget: 0.5` for gradient checkpointing.

7. **Debug on CPU first:** Set `compile: false` and use a small subset of data. GPU debugging is painful.

## Troubleshooting

### Common Issues

**"CUDA out of memory"**
- Reduce `batch_size` in config
- Set `mem_budget: 0.5` (or lower) for gradient checkpointing
- Disable `compile` (compiled models use more memory initially)

**NaN losses**
- Disable AMP: `amp: false`
- Lower learning rate
- Add gradient clipping: `max_grad_norm: 1.0`
- Check your data for NaN/Inf values

**torch.compile errors**
- Set `compile: false` - not all models compile successfully
- Check for dynamic shapes in your model (variable sequence lengths, etc.)
- Compile works best with static shapes

**Training doesn't resume from checkpoint**
- Verify `checkpoint_name: latest` is set in config
- Check that `latest.pt` exists in your results directory
- Make sure `restart: true` if you want to continue training

**WandB not logging**
- Check `WANDB_API_KEY` in `.env`
- Verify `wandb.enabled: true` in config
- Run `wandb login` manually to test

**"Address already in use" with torchrun**
- A previous run didn't clean up properly
- Kill orphan processes: `pkill -f torchrun`
- Or wait a minute for the port to be released

**SLURM job immediately fails**
- Check output file in `results/00_slrm_logs/`
- Verify paths in `.env` are absolute paths
- Make sure the venv exists: `ls $BASE_DIR/.venv/bin/python`

### Debugging Tips

**Test your model independently:**
```python
# In a Python shell
from ml_suite.models.model_utils import get_model
config = {"type": "my_model", "input_dim": 128}
model = get_model(config)
x = torch.randn(2, 128)  # Small batch
y = model(x)
print(y.shape)  # Verify output shape
```

**Test your dataset:**
```python
from ml_suite.data.dataset import get_dataset
config = {"name": "my_dataset"}
ds = get_dataset(config, split="train")
x, y = ds[0]
print(x.shape, y.shape)  # Verify shapes match model expectations
```

**Run a quick sanity check:**
```bash
# Single GPU, small run
python run_training.py --config_path config.yml
# Set total_updates: 100 in config for a quick test
```

## Glossary

| Term | Meaning |
|------|---------|
| **AMP** | Automatic Mixed Precision - using float16/bfloat16 for faster training |
| **Batch size** | Number of samples processed before a gradient update |
| **Checkpoint** | Saved state of model, optimizer, scheduler for resumption |
| **DDP** | DistributedDataParallel - PyTorch's multi-GPU training wrapper |
| **Epoch** | One complete pass through the training dataset |
| **Gradient clipping** | Limiting gradient magnitude to prevent exploding gradients |
| **GradScaler** | Scales loss in AMP to prevent underflow in float16 |
| **HPC** | High-Performance Computing cluster |
| **LR** | Learning Rate |
| **NCCL** | NVIDIA Collective Communications Library - for GPU-to-GPU communication |
| **SLURM** | Job scheduler used on HPC clusters |
| **torchrun** | PyTorch's launcher for distributed training |
| **Update** | One gradient update step (forward + backward + optimizer step) |
| **Warmup** | Gradually increasing LR at start of training for stability |
| **World size** | Total number of GPUs/processes in distributed training |

## Learning Resources

- [PyTorch Distributed Training Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Automatic Mixed Precision](https://pytorch.org/docs/stable/amp.html)
- [torch.compile documentation](https://pytorch.org/docs/stable/torch.compiler.html)
- [WandB Quickstart](https://docs.wandb.ai/quickstart)
- [SLURM Documentation](https://slurm.schedmd.com/documentation.html)
