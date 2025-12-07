# KPL: Knowledge-enhanced Proxy Learning for Training-Free Medical Image Classification

This repository contains the official implementation of the paper "KPL: Knowledge-enhanced Proxy Learning for Training-Free Medical Image Classification".

## Introduction

KPL is a training-free method for medical image classification that leverages knowledge-enhanced proxies. It utilizes CLIP's text encoder to generate text proxies and optimizes visual proxies using Optimal Transport (Greenkhorn algorithm).

<img width="772" height="322" alt="image" src="https://github.com/user-attachments/assets/e72d57f0-8a53-4a11-99a0-7f2a99b541f0" />

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- clip
- numpy
- scipy
- matplotlib
- sklearn

## Usage

### Data Preparation

Please organize your dataset in the standard ImageFolder structure:
```
/path/to/dataset/
    val/
        class1/
            img1.jpg
            ...
        class2/
            img2.jpg
            ...
```

### Running the Code

To run the classification on a specific dataset, use the following command:

```bash
python Main.py --data_path /path/to/dataset --type <dataset_type> --arch ViT-L/14@336px
```

### Arguments

- `--data_path`: Path to the dataset directory.
- `--type`: Type of the dataset (e.g., `brain`, `eyes`, `Lung`, `imagenet`, `cub`, `foods`, `pets`, `place`, `cataract`, `Cell`).
- `--arch`: Model architecture (default: `ViT-L/14@336px`). Choices: `RN50`, `RN101`, `ViT-B/32`, `ViT-B/16`, `ViT-L/14`, `ViT-L/14@336px`.
- `--k`: Number of top-k text features to select (default: 8).
- `--iters_proxy`: Number of iterations for learning vision proxy (default: 2000).
- `--iters_sinkhorn`: Number of iterations for optimizing Sinkhorn distance (default: 20).
- `--lr`: Initial learning rate (default: 10).
- `--batch-size`: Mini-batch size (default: 256).

### Example

```bash
python Main.py --data_path ./data/brain_tumor --type brain --k 8
```

## Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@inproceedings{liu2025kpl,
  title={Kpl: Training-free medical knowledge mining of vision-language models},
  author={Liu, Jiaxiang and Hu, Tianxiang and Du, Jiawei and Zhang, Ruiyuan and Zhou, Joey Tianyi and Liu, Zuozhu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={18},
  pages={18852--18860},
  year={2025}
}
```
