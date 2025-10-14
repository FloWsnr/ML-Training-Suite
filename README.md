# ML Training and Inference Suite

This repository contains a suite of tools and scripts for training and inference of machine learning models.
The suite includes functionalities for data preprocessing, model training, evaluation, and deployment.

## Features
- Modular design for easy integration and extension
- torch.compile (with memory constraints)
- Automatic Mixed Precision (AMP) support
- Distributed training capabilities
- WandB integration for experiment tracking
- Configurable via YAML files
- Checkpointing and resuming training support
- Time keeping for graceful shutdowns and resuming on HPCs

## Structure

The main part of the suite is located in the train directory.
- `eval.py` and `train_base.py`: Main scripts for training and evaluation. They are kept as pure python code without any CLI args / config parsing to allow easy integration into other projects. They are also testable via pytest.
- `run_training.py`: CLI script for training. Ugly parsing and setup code goes here. Not easy to test.
- `utils/`: Utility functions for checkpointing, logging, time keeping, LR scheduling, etc.
- `train.yml`: Example config file for training. Modify as needed.
- `scripts/train.sh`: Example bash script for running training on a local machine or HPC. Modify as needed. Calls the run_training.py script.


## Instructions
1. Fork / copy the repository
2. Install miniforge or anaconda if you don't have it already.
3. Install the required dependencies in a conda environment. Of course, also install your own dependencies if needed, you don't have to use conda but some form of environment management is recommended.

```bash
conda create -n train_env python=3.13
conda activate train_env
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install pyyaml python-dotenv pytest wandb einops
pip install -e .
```

3. Create a ``.env`` file with necessary environment variables in the top level directory. We do this to avoid hardcoding paths and sensitive information in the code. This way, you can use different .env files for different setups (local, HPC, etc.). Also, you can easily publish your code without exposing sensitive information. The base dir and results dir are used in the sh script to start the run. Use the data dir in your dataset code.

```bash
WANDB_API_KEY=your_api_key_here
BASE_DIR=/path/to/your/base/dir # path to this repo
DATA_DIR=/path/to/your/data/dir # path to where your data is stored, should be reused
RESULTS_DIR=/path/to/your/results/dir # checkpoints and wandb files are stored here
```

4. (Optional) Set up your [WandB account](https://wandb.ai/site/models/) and project for experiment tracking. Highly recommended!
5. Add your model code in the models directory and update the ``get_model`` function in `ml_suite/models/model_utils.py`. Currently, the function gets a config dictionary and returns a model instance. You can change this as needed.
6. Create your dataset code. Make sure the dataset returns a tuple (input, target) for each item or change the code in train and eval.
7. Check out the loss functions `ml_suite/models/loss_fns.py` and add missing ones.

8. Create a dir for your results. The slurm script currently assumes that the results dir is in the top level dir (i.e. the same dir as the .env file). You can change this as needed.

9. Copy the config file "ml_suite/train/train.yml" to your results dir and modify it as needed. We copy the config file to the results dir to keep track of the exact config used for each run. Important things to change:
- dataset params
- Make sure to set the correct model params
- check if your model can be compiled with torch.compile
- training params (batch size, learning rate, number of updates, etc.)
- use the correct wandb project name and entity

10. Run the vphysics/train/scripts/train_riv.sh
- Make sure set the correct conda env name here or the python path if not using conda
-

## Notes

- The training script uses number of gradient updates (number of batches) instead of epochs to determine training duration. This is to ensure consistent training time across different dataset sizes and batch sizes. The config parameter `updates_per_epoch` controls how many updates are done per evaluation cycle. If you want to train "x number of epochs", you can either calculate the respective number of updates or adapt the code.
- Similarly, the LR schedulers assume updates, not epochs as steps.
- The script supports resuming training from a checkpoint. If a checkpoint is found in the results directory, training will resume from the latest checkpoint. You can also specify a specific checkpoint to resume from in the config file. If no checkpoint is found, training will start from scratch.
- Automatic Mixed Precision (AMP) is enabled by default for faster training. You can disable it in the config file if needed. Not all models support AMP. Same thing for torch.compile. Use the ``mem_budget`` parameter to control memory usage. With this, you can run a model with a larger batch size than your GPU memory would normally allow at the cost of some additional computation time.
- The suite includes chained LR schedulers (up to three). You can configure them in the config file. The first scheduler is applied first for x number of updates, followed by the second, and then the third. They also work on restarting from a checkpoint.
- Use the ``restart`` config parameter to decide wether training is fully continued (LR, optimizer), or whether only the model weights are loaded and a new configuration is used, i.e. finetuning, changing of LR scheduler etc.


## Suggestions for starters

- Try AMP and torch.compile, they can give significant speedups with minimal code changes
- Use WandB for tracking experiments, it provides a lot of useful features for monitoring and comparing experiments
- Use the checkpointing feature to save and resume training, especially for long-running jobs
- Linear warmup followed by cosine annealing is a good default choice for LR scheduling
- For unstable training, try gradient clipping ``max_grad_norm`` in the config file
