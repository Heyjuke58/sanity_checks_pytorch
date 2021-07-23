# Sanity Checks for Saliency Methods

In this repo we reimplement a sanity check (cascading randomization) for numerous saliency methods. We let us guide by a paper from [Adebayo et al.](https://arxiv.org/abs/1810.03292) and their [code](https://github.com/adebayoj/sanity_checks_saliency).

We examine the sanity check using a basic CNN trained on MNIST as well as a pre-trained ResNet-18 on ImageNet. You can find our detailed results in the PDF file.

How to get the code running:

1. clone repo with submodule

`git clone https://github.com/Heyjuke58/sanity_checks_pytorch.git --recurse-submodules`

2. setup environment via conda

`conda env create --file environment.yml`
