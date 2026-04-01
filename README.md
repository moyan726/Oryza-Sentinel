# Rice Leaf Disease Classification Course Project

This repository turns the bundled rice leaf disease image dataset into a reproducible deep learning course project.

## Goals

- Build a four-class rice leaf disease classifier.
- Report both the official split result and a de-duplicated clean split result.
- Support transfer learning, hyperparameter tuning, plots, report assets, and defense materials.

## Project Layout

- `Rice Leaf Disease Images/`: original dataset
- `configs/`: experiment YAML configs
- `docs/`: Chinese report and defense outline
- `outputs/`: generated manifests, models, figures, metrics, and summaries
- `src/rice_leaf_disease/`: reusable package code
- `prepare_dataset.py`: audit dataset and generate manifests
- `train.py`: train one experiment
- `tune.py`: tune EfficientNet-B0 hyperparameters with Optuna
- `evaluate.py`: evaluate checkpoints and generate plots
- `predict.py`: single-image inference demo

## Recommended Environment

1. Create the conda environment:

   ```powershell
   conda env create -f environment.yml
   conda activate rice-leaf-dl
   ```

2. Confirm PyTorch can see the GPU:

   ```powershell
   python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
   ```

## Quick Start

1. Prepare manifests and dataset audit assets:

   ```powershell
   python prepare_dataset.py --dataset-root "Rice Leaf Disease Images"
   ```

2. Run a smoke experiment:

   ```powershell
   python train.py --config configs/smoke_official.yaml
   ```

3. Train the main official-split model:

   ```powershell
   python train.py --config configs/official_efficientnet.yaml
   ```

4. Evaluate a checkpoint:

   ```powershell
   python evaluate.py --config configs/official_efficientnet.yaml --checkpoint outputs/runs/official_efficientnet_b0/best_model.pt
   ```

5. Run hyperparameter tuning:

   ```powershell
   python tune.py --config configs/official_efficientnet.yaml
   ```

## Notes

- The official split contains duplicate images across splits. Use the `clean` view for stricter evaluation.
- Training automatically falls back to CPU if CUDA is unavailable.
- Reports and defense materials are written in Chinese under `docs/`.
