This is the repository associated with the paper [*A path-norm toolkit for modern networks: consequences, promises and challenges*](https://arxiv.org/abs/2310.01225) by [Antoine Gonon](https://agonon.github.io/), [Nicolas Brisebarre](https://perso.ens-lyon.fr/nicolas.brisebarre/index.html.en), [Elisa Riccietti](https://perso.ens-lyon.fr/elisa.riccietti/) and [Rémi Gribonval](https://people.irisa.fr/Remi.Gribonval/), ICLR 2024.
- [Setup with conda](#setup-with-conda)
- [Reproduction of the results in [GBRG24]](#reproduction)
- [Functions of interest](#functions-of-interest)
- [System information](#system-information)
- [Acknowledgement and third-party notice](#acknowledgement-and-third-party-notice)
- [License](#license)
- [References](#references)

# Setup with Conda

Follow these steps to set up the project environment using conda:

1. Clone the repository:
   ```bash
   git clone https://github.com/agonon/pathnorm_toolkit.git
   cd pathnorm_toolkit
   ```

2. Create a conda environment:
   ```bash
   conda create --name your_env_name
   ```

3. Activate the conda environment:
   ```bash
   conda activate your_env_name
   ```

4. Install pip within the environment:
   ```bash
   conda install pip
   ```

5. Install the project dependencies:
   ```bash
   pip install -e .
   ```

To deactivate and remove the environment when you're done:
```bash
conda deactivate
conda env remove --name your_env_name
```

# Reproduction of Results in [GBRG24](#GBRG24)<a name="reproduction"></a>

We now describe the scripts to reproduce the different results of [GBRG24](#GBRG24). Whenever ImageNet is needed, precomputed results are available to save users from having to perform the computation from scratch. Options to perform the computations from scratch using ImageNet are described below and in each bash script.

## Table 2 (1st part of the bound of Theorem 3.1)

### Option 1: Use Pre-computed Results
- *Time on my machine:* 3 seconds.

If you decide to use the pre-computed value of B (max $L^\infty$ norm of the input images of ImageNet, normalized for inference), simply run:
```bash
bash scripts/0_compute_1st_part_bound_resnets.sh
```

### Option 2: Recompute Using ImageNet
- *Time on my machine:* 10 minutes on a single GPU V100 16GB.
- *Memory requirements (can vary depending on your filesystem):* 165GB (for ImageNet).

If you decide to recompute the value of B:
1. Open `scripts/0_compute_1st_part_bound_resnets.sh` in an editor.
2. Set `data_dir` to the ImageNet directory.
3. Adjust `workers` and `batch_size` variables.
4. Comment the line starting with `--B`.
5. Run `bash scripts/0_compute_1st_part_bound_resnets.sh`.

## Tables 3, 5, and Figure 3 (path-norms and margins of pre-trained ResNets)

### Option 1: Use Pre-computed Results
- *Time on my machine:* 40 seconds.
- *Memory requirements (can vary depending on your filesystem):* 0.7GB on your home directory used to download pre-trained ResNets available on Torch at the beginning of the script.

If you decide to use the pre-computed margins of all images of ImageNet for the different pre-trained ResNets of Torch, simply run:
```bash
bash scripts/1_compute_pretrained_pathnorm_margins_resnets.sh
```

### Option 2: Recompute Using ImageNet
- *Time on my machine:* 2 hours on a single GPU V100 16GB.
- *Memory requirements (can vary depending on your filesystem):* 165GB (for ImageNet), additional 0.7GB on your home directory used to download pre-trained ResNets available on Torch at the beginning of the script.

If you decide to recompute the margins of pre-trained ResNets on ImageNet:
1. Open `scripts/1_compute_pretrained_pathnorm_margins_resnets.sh` in an editor.
2. Set `data_dir` to the ImageNet directory.
3. Set `saving_dir` to where you want to save the results, ensuring at least 25MB free memory.
4. Adjust `workers` and `batch_size` variables.
5. Comment the line `--margins-already-computed \`.
6. Run `bash scripts/1_compute_pretrained_pathnorm_margins_resnets.sh`.


## Figure 4 (path-norms of sparse networks)

### Option 1: Use Pre-computed Results
- *Time on my machine:* 1 minute 20 seconds.

If you decide to use the pre-computed metrics (path-norms, accuracy...) of a ResNet18 trained for 20 IMP iterations on ImageNet, simply run:
```bash
bash scripts/3_plot_imp.sh
```

### Option 2: Recompute Using ImageNet
- *Time on my machine:*  20 minutes per epoch (=~10mins/epoch of gradient update, and ~10mins/epoch to compute and save data margins after gradient update) on a single GPU V100 16GB, resulting in 30 hours per IMP iteration of 90 epochs and a total of **25 days** for 20 IMP iterations of 90 epochs.
- *Memory requirements (can vary depending on your filesystem):* 165GB (for ImageNet), additional 0.2GB per IMP iteration for saving metrics and weights at best validation top-1 accuracy during training (4GB if you stay with the default 20 IMP iterations).

If you decide to retrain a ResNet18 for 20 iterations of IMP (Iterative Magnitude Pruning) on ImageNet :
1. Open `scripts/2_train_imp.sh` in an editor.
2. Set `data_dir` to the ImageNet directory.
3. Set `saving_dir` to where you want to save the results, ensuring at least 0.2GB free memory per IMP iteration.
4. Adjust `workers` and `batch_size` variables.
5. Run `bash scripts/2_train_imp.sh`.

To generate plots with your new results, open `scripts/3_plot_imp.sh`, set `results_training_dir` to match the `saving_dir` used in `scripts/2_train_imp.sh`, and run `bash scripts/3_plot_imp.sh`.

## Figure 5 (path-norms with increased train size)

### Option 1: Use Pre-computed Results
- *Time on my machine:* 3 seconds.

If you decide to use the pre-computed path-norms of a ResNet18 trained on 5 increasing subsets of ImageNet, simply run:
```bash
bash scripts/5_plot_increasing_dataset.sh
```

### Option 2: Recompute Using ImageNet
- *Time on my machine:* ~10/k minutes per epoch on a single GPU V100 16GB for 1/k of the full dataset. By default, the script trains for 90 epochs, on 1/2, 1/4, 1/8, 1/16 and 1/32 of the full dataset, each for 3 different seeds, resulting in a total of approximately **44 hours**.
- *Memory requirements (can vary depending on your filesystem):* 165GB (for ImageNet), additional 0.33GB per seed and per training size for saving metrics and weights at best validation top-1 accuracy during training (5GB if you stay with the default 3 seeds and 5 training sizes).

If you decide to retrain a ResNet on increasing subsets of ImageNet:
1. Open `scripts/4_train_increasing_dataset.sh` in an editor.
2. Set `data_dir` to the ImageNet directory.
3. Set `saving_dir` to where you want to save the results, ensuring at least 0.33GB free memory per seed and per training size.
4. Adjust `workers` and `batch_size` variables.
5. Run `bash scripts/4_train_increasing_dataset.sh`.

To generate plots with your new results, open `scripts/5_plot_increasing_dataset.sh`, set `results_training_dir` to match the `saving_dir` used in `scripts/4_train_increasing_dataset.sh`, and run `bash scripts/5_plot_increasing_dataset.sh`.


# Functions of Interest

Here are key functions within the project:

- **Computing the Path-Norm of a model (see [ok_models.json](https://github.com/agonon/pathnorm_toolkit/blob/main/ok_models.json) for valid ones):**
  - Function: `get_path_norm`
  - Location: `src/pathnorm/path_norm/compute_path_norm.py`
  - Example of use: refer to `scripts/compute_path_norm_models.py`

- **Computing the First Part of the Bound of Theorem 3.1 in [GBRG24](#GBRG24) for a ResNet:**
  - Script: `src/pathnorm/path_norm/compute_1st_part_bound_resnets.py`
  - Example of use: refer to `scripts/0_compute_1st_part_bound_resnets.sh`

- **Computing the Margins of a Model:**
  - Function: `get_all_margins`
  - Location: `src/pathnorm/path_norm/compute_margin.py`
  - Example of use: refer to `src/pathnorm/path_norm/pathnorm_and_margins_pretrained_resnets.py`

- **Train a ResNet with IMP:**
  - Script: `src/pathnorm/train_imagenet.py`
  - Example of use: refer to `scripts/2_train_imp.sh`

These functions and scripts play a crucial role in various aspects of the project, from computing path-norms and model margins to training ResNets with IMP. Refer to the provided examples for correct usage and integration within your own workflows.

# System Information

This code has been tested on a Linux system (February 2024) with the following hardware configuration:

- CPU: Intel(R) Xeon(R) Silver 4215R @ 3.20GHz
- GPU: NVIDIA A100 PCIe 40GB, NVIDIA V100 PCIe 16GB

## Package Versions at the Time of Testing:
```
conda 24.1
python 3.12
torch 2.2
torchvision 0.17
nvidia-cuda 12.1
numpy 1.26
pandas 2.2
matplotlib 3.8
```

# Acknowledgement and third-party notice

The backbone of [src/path_norm/train_imagenet.py](<src/pathnorm/train_imagenet.py>) (excluding stuff related to path-norms, margins, and iterative pruning) is an adaptation of the [PyTorch example](https://github.com/pytorch/examples/blob/main/imagenet/main.py) made in collaboration with [Léon Zheng](https://leonzheng2.github.io/). Please review the [NOTICE](<NOTICE.md>) file for information about the license of the [PyTorch example](https://github.com/pytorch/examples/blob/main/imagenet/main.py).

# License
This project is licensed under the [BSD 3-Clause License](https://opensource.org/license/BSD-3-clause), see the [LICENSE](<LICENSE>) file for details.

# References
[GBRG24]:<a name="GBRG24"></a> Gonon, A., Brisebarre, N., Riccietti, E., & Gribonval, R. [*A path-norm toolkit for modern networks: consequences, promises and challenges.*](https://arxiv.org/abs/2310.01225) ICLR Spotlight 2024.
