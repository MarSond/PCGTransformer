# PCG Classification Framework

A framework for developing and evaluating PCG (Phonocardiogram) classification models using CNNs and BEATs transformers.

## Project Structure

```
├── data/                    # Dataset storage (not included in repo)
│   ├── physionet2016/      # PhysioNet 2016 Challenge dataset 
│   └── physionet2022/      # PhysioNet 2022 Challenge dataset
├── documents/              # Technical documentation
├── MLHelper/              # Core framework modules
│   ├── audio/             # Audio processing and augmentation
│   │   ├── augmentation.py
│   │   ├── preprocessing.py
│   │   └── audioutils.py 
│   ├── metrics/           # Metrics calculation and tracking
│   │   ├── metrics.py     # MetricsTracker implementation
│   │   └── loss.py
│   ├── tools/             # Common utilities
│   │   ├── utils.py       # General utility functions
│   │   └── logging_helper.py
│   ├── constants.py       # Global constants and configurations
│   ├── config.py         # Configuration management
│   ├── dataset.py        # Dataset base classes
│   └── ml_loop.py        # Training loop implementation
├── runs/                 # Training run outputs
├── final_runs/           # Production model runs
├── temp_code/           # Helper scripts and experiments
├── run.py               # Main run class (TODO: Move to MLHelper)
├── build_dataset.py     # Dataset parsing and preprocessing
└── project_config.py    # Project-specific configuration
```

See project_config.pym run.py and start_training.py for main configurations and entry points. 

## Setup

1. Create environment using Mamba/Conda:
```bash
conda env create -f documents/conda_env.yml
conda activate transformer
```


### Training

- `start_training.py`: Main training script with predefined configurations
  - Contains validated configurations for different model architectures
  - Use for production training runs

### Optimization

- `start_optim.py`: Hyperparameter optimization for CNN/BEATs models
  - Uses Optuna for automated parameter search
  - Supports both fixed-length and heart cycle chunking

- `start_optim_embedding.py`: Optimization focused on embedding classifiers
  - Tunes UMAP, kNN and other embedding parameters
  - Specialized for BEATs feature extraction

- `start_optim_embedding_finetune.py`: Fine-tuning of embedding models
  - For detailed parameter adjustments after coarse optimization

### Testing & Validation

- `start_benchmark.py`: Performance benchmarking of trained models
  - Measures inference speed and memory usage
  - Compares different model architectures

- `start_inference.py`: Standalone inference script
  - Load trained models and run predictions
  - Export results in standard format

- `start_looptest.py`: Framework testing script
  - Validates new features and metrics
  - Quick iteration testing with minimal configurations

## Running Training

1. Basic training with default configuration:
```bash
python start_training.py
```

2. View training results:
```bash
python RunMetricsParser.py
```

## Project Features

- Modular architecture supporting multiple model types
- Comprehensive metrics tracking and visualization
- Automatic k-fold cross validation
- Configurable data augmentation
- Support for both fixed-length and cycle-based audio chunking
- Integration with Optuna for hyperparameter optimization


