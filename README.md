# CatPred: A comprehensive framework for deep learning in vitro enzyme kinetic parameters kcat, Km and Ki

[![DOI](https://img.shields.io/badge/DOI-10.1101/2024.03.10.584340-blue)](https://www.biorxiv.org/content/10.1101/2024.03.10.584340v2)
[![Colab](https://img.shields.io/badge/GoogleColab-tiny.cc/catpred-red)](https://tiny.cc/catpred)


## Table of Contents

- [System Requirements](#requirements)
- [Installation](#installation)
  * [Installing from source](#option-2-installing-from-source)
- [Google Colab Interface](#web-interface)

## System Requirements

For using pre-trained models to predict, any machine running a Linux based operating system is recommended.
For training, we recommend using a Linux based operating system on a GPU-enabled machine.

Both training and prediction have been tested on Ubuntu 20.04.5 LTS with NVIDIA A10 and CUDA Version: 12.0

To train with GPUs, you will need:
 * cuda >= 11.7
 * cuDNN

## Installation

Both options require conda, so first install Miniconda from [https://conda.io/miniconda.html](https://conda.io/miniconda.html).

Then proceed to either option below to complete the installation. If installing the environment with conda seems to be taking too long, you can also try running `conda install -c conda-forge mamba` and then replacing `conda` with `mamba` in each of the steps below.

**Note for machines with GPUs:** You may need to manually install a GPU-enabled version of PyTorch by following the instructions [here](https://pytorch.org/get-started/locally/). If you're encountering issues with not using a GPU on your system after following the instructions below, check which version of PyTorch you have installed in your environment using `conda list | grep torch` or similar. If the PyTorch line includes `cpu`, please uninstall it using `conda remove pytorch` and reinstall a GPU-enabled version using the instructions at the link above.

### Installing from source (~5 mins)

1. `git clone https://github.com/maranasgroup/catpred.git`
2. `cd catpred`
3. `conda env create -f environment.yml`
4. `conda activate catpred`
5. `pip install -e .`


## Web Interface

For ease of use without any hardware requirements, a Google Colab interface is available here: [tiny.cc/catpred](http://tiny.cc/catpred).
It contains sample data, instructions and installation all in the Colab notebook.

## Reproducing publication training/results

In order to train publication models, you must download and extract training datasets using
```
wget https://https://catpred.s3.amazonaws.com/publication_training_datasets.tar.gz -q
tar -xzf publication_training_datasets.tar.gz
```

## Training

To train a model, run:
```
```

## License <a name="license"></a>

This source code is licensed under the MIT license found in the `LICENSE` file
in the root directory of this source tree.

## Citations <a name="citations"></a>

If you find the models useful in your research, we ask that you cite the relevant paper:

```bibtex
@article {Boorla2024.03.10.584340,
	author = {Veda Sheersh Boorla and Costas D. Maranas},
	title = {CatPred: A comprehensive framework for deep learning in vitro enzyme kinetic parameters kcat, Km and Ki},
	elocation-id = {2024.03.10.584340},
	year = {2024},
	doi = {10.1101/2024.03.10.584340},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/03/26/2024.03.10.584340},
	eprint = {https://www.biorxiv.org/content/early/2024/03/26/2024.03.10.584340.full.pdf},
	journal = {bioRxiv}
}
```
