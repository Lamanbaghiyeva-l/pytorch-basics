# PyTorch Basics

My hands-on path from a hand-written training loop to a CNN that reaches 99% on MNIST.
Every script here was written by me while learning PyTorch from scratch.

## Projects

| File | What it does | Result |
|---|---|---|
| `01_tensors.py` | Tensor basics: creation, shape/dtype/device, indexing, `*` vs `@`, moving to GPU | — |
| `02_linear_regression_lr_sweep.py` | Learns `y = 2x + 1` from 5 points with `nn.Linear`; compares learning rates and optimizers | w ≈ 2.000, b ≈ 1.000 |
| `03_mnist_mlp.py` | Fully connected network (784 → 256 → 64 → 10) on MNIST | **98.08%** test accuracy |
| `04_mnist_cnn.py` | CNN (2 × Conv2d + MaxPool, then Linear) on MNIST | **99.06%** test accuracy |

## Results on MNIST

| Model | Parameters | Epochs | Optimizer | Test accuracy |
|---|---|---|---|---|
| MLP (128 hidden) | 109,386 | 3 | Adam, lr 0.001 | 96.96% |
| MLP (256 hidden) | 218,058 | 10 | Adam, lr 0.001 | 98.08% |
| **CNN** | **206,922** | 10 | Adam, lr 0.001 | **99.06%** |

Learning rate sweep (Adam): lr 0.001 worked best for both models; lr 0.1 and 1 collapsed to ~10% accuracy (random guessing, loss stuck at ~2.30).

## What I learned

- **The training loop is always the same five steps:** `zero_grad` → forward → loss → `backward` → `step`. Only the model and data change.
- **Learning rate matters more than I expected.** On linear regression, SGD with lr 0.1 exploded, while Adam stayed stable even with large learning rates. On MNIST, the same large learning rates destroyed the model, so the best value depends on the problem and has to be tested.
- **A sweep needs a fresh model for each setting.** My first sweep reused one model, so later runs started from already-trained weights and the comparison was unfair.
- **CNNs beat MLPs on images with fewer parameters.** The two Conv layers have under 5,000 parameters, but because each filter slides over the whole image and reuses its weights, the CNN halved the errors of the MLP.
- **Shapes and devices cause most bugs.** `nn.Linear` needs `[batch, features]`, the Linear layer after Flatten needs `channels × height × width` inputs, and the model and data must be on the same device.
- **Environment setup is part of the job.** The server's NVIDIA driver supports CUDA 12.6, so I had to install a matching PyTorch build to use the GPU.

## How to run

```bash
pip install torch torchvision
python 04_mnist_cnn.py
```

MNIST downloads automatically into `data/` on the first run. Tested with PyTorch 2.8 (CUDA 12.6) on an NVIDIA RTX 6000 Ada.

