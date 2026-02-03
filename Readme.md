This repository provides the code for the paper *Shaky Prepend: A Multi-Group Learner with Improved Sample Complexity*

## Contents
- `src/prepend.py`: Prepend baselines (constant and group-weighted epsilon).
- `src/shaky_prepend.py`: Shaky prepend variants with Laplace noise.
- `src/sleeping_expert.py`: Sleeping-expert style baseline.
- `simulation_experiment.py`: Data generators, experiment runners, and plotting.
- `utils.py`: Loss helpers and plotting utilities.
- `results/`: Example figures produced by the experiments.

## Setup
### Option A (recommended; portable)
```sh
conda env create -f environment.yml
conda activate shaky-prepend
```

### Option B (exact pinned environment; macOS arm64)
```sh
conda create --name shaky-prepend --file requirements.txt
conda activate shaky-prepend
```

## Run experiments
```sh
python simulation_experiment.py
```

Plots are saved under `results/`.
