# SNN-Balancing

Our implementation for ICJNN 2015 paper "Fast-Classifying, High-Accuracy Spiking Deep Networks Through Weight and Threshold Balancing"

## Setup

```shell
conda create -n SNN python=3.8
conda activate SNN
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Files

1. `main.py` is the main source code file that replicates the original paper. The execution log can be found at `result.log`, and the output is written to `result.json` and `result.png`.
2. `main_fashion_mnist.py` is another source code file that augments the original paper using the Fashion MNIST dataset. The execution log can be found at `result-fashion.log`, and the output is written to `result-fashion.json` and `result-fashion.png`.

## Results

See `REPORT.pdf`.
