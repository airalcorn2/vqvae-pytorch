# PyTorch VQ-VAE

[![Open VQ-VAE in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1n6rPRkon0Bc-PyrXwDDphlRjrqwdV9jd)

This is a minimal PyTorch implementation of the VQ-VAE model described in "[Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)".
I tried to stay as close to [the official DeepMind implementation](https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb) as possible while still being PyTorch-y, and I tried to add comments in the code referring back to the relevant sections/equations in the paper.

To train the model on the CIFAR-10 dataset using the same hyperparameters described in the paper, run:

```bash
python3 train_vqvae.py
```

It should only take a few minutes on a modern GPU (a Colab notebook can be found [here](https://colab.research.google.com/drive/1n6rPRkon0Bc-PyrXwDDphlRjrqwdV9jd?usp=sharing)).
After training, the script saves the following two images:

**Validation Set Samples**

![](true.jpg)

**Reconstructions**

![](recon.jpg)
