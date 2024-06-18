# Low-Light Image Denoising with Residual Learning

This repository contains the implementation of a deep neural network for denoising extreme low-light images using residual learning. The proposed method leverages a novel network architecture incorporating Leaky ReLU activation functions and Squeeze-and-Excitation blocks to achieve significant improvements in noise reduction and color accuracy.

## Overview

Imaging in low-light conditions is challenging due to low photon count and low signal-to-noise ratio (SNR). Traditional denoising and enhancement techniques often fall short in extreme conditions. This project introduces a deep learning-based approach for end-to-end denoising of low-light images.

## Methodology

The network architecture is built on a residual learning framework that focuses on learning the difference between low-light and high-quality reference images. Key components of the architecture include:

- **Residual Learning Framework**: Simplifies the learning process by focusing on residuals rather than entire images.
- **Leaky ReLU Activation**: Preserves more information by allowing small gradients for negative values.
- **Squeeze-and-Excitation Blocks**: Enhances feature representation and network performance.
- **Constant Feature Size**: Avoids downscaling to maintain image details and reduce complexity.

## Dataset

The dataset used for training and evaluation consists of 485 low-light images paired with high-quality reference images provided by VLG IITR. The images include various real-world low-light scenarios.

## Results

This network demonstrated significant improvements in noise reduction while preserving color and texture details at the same time reducing computational cost. The performance was evaluated using metrics such as Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM).

- **PSNR**: 18.5
- **SSIM**: 0.775

## Installation

Clone the repository and install the required dependencies.

```bash
git clone https://github.com/username/repo.git
cd repo
```

## Note

### Google Colab
- If this code is run on colab then no extra installation is need. Although this code works on both CPU and GPU, prefer GPU.

### Local or Remote System
- create an environment
- For linux
    ```bash
    conda create --name denoise python=3.8
    conda activate denoise
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    conda install scikit-image
    conda install matplotlib
    ```
- for other os refer to `https://pytorch.org/get-started/locally/` for creating compatible GPU enabled enviroment. 
