# CatPred: A comprehensive framework for deep learning in vitro enzyme kinetic parameters kcat, Km and Ki

[![DOI](https://img.shields.io/badge/DOI-10.1101/2024.03.10.584340-blue)](https://www.biorxiv.org/content/10.1101/2024.03.10.584340v2)
[![Colab](https://img.shields.io/badge/GoogleColab-tiny.cc/catpred-red)](https://tiny.cc/catpred)

## Table of Contents

- [Google Colab Interface](#web-interface)
- [Local Installation](#local-demo)
- [Reproducibility](#reproduce)
- [Acknowledgements](#acknw)
- [License](#license)

## Google Colab Interface Demo (easy) <a name="web-interface"></a>

For ease of use without any hardware requirements, a Google Colab interface is available here: [tiny.cc/catpred](http://tiny.cc/catpred).
It contains sample data, instructions and installation all in the Colab notebook.

## Local Installation <a name="local-demo"></a>

If you would like to install the package on a local machine, please follow the following instructions.

### System Requirements <a name="requirements"></a>

For using pre-trained models to predict, any machine running a Linux based operating system is recommended.
For training, we recommend using a Linux based operating system on a GPU-enabled machine.

Both training and prediction have been tested on Ubuntu 20.04.5 LTS with NVIDIA A10 and CUDA Version: 12.0

To train or predict with GPUs, you will need:
 * cuda >= 11.7
 * cuDNN

### Installation <a name="installing"></a>

Both options require conda, so first install Miniconda from [https://conda.io/miniconda.html](https://conda.io/miniconda.html).

Then proceed to either option below to complete the installation. If installing the environment with conda seems to be taking too long, you can also try running `conda install -c conda-forge mamba` and then replacing `conda` with `mamba` in each of the steps below.

**Note for machines with GPUs:** You may need to manually install a GPU-enabled version of PyTorch by following the instructions [here](https://pytorch.org/get-started/locally/). If you're encountering issues with not using a GPU on your system after following the instructions below, check which version of PyTorch you have installed in your environment using `conda list | grep torch` or similar. If the PyTorch line includes `cpu`, please uninstall it using `conda remove pytorch` and reinstall a GPU-enabled version using the instructions at the link above.

#### Installing and downloading pre-trained models (~5 mins)
1. `mkdir catpred_pipeline`
2. `wget https://catpred.s3.amazonaws.com/capsule_data.tar.gz -q`
3. `tar -xzf capsule_data.tar.gz`
4. `git clone https://github.com/maranasgroup/catpred.git`
5. `cd CatPred`
6. `conda env create -f environment.yml`
7. `conda activate catpred`
8. `pip install -e .`

## Reproducing publication results <a name="reproduce"></a>

We provide three separate ways for reproducing the results of the publication. 

### 1. Quick method: 

Estimated run time: Few minutes

Can be run using 
`./reproduce_quick.sh`

For all results pertaining to CatPred, UniKP, DLKcat and Baseline models, this method only uses pre-trained predictions and analyses to reproduce results of the publications including all main and supplementary figures. 

### 2. Prediction method: 

Estimated run time: Upto a day depending on your GPU

Can be run using 
`./reproduce_prediction.sh`

For results pertaining to CatPred, this method uses pre-trained models to perform predictions on test sets. 
For results pertaining to UniKP, DLKcat and Baseline, this method uses only uses pre-trained predictions and analyses to reproduce results of the publications including all main and supplementary figures. 

### 3. Training method: 

Estimated run time: Upto 12-14 days depending on your GPU

Can be run using 
`./reproduce_training.sh`

For all results pertaining to CatPred, UniKP, DLKcat and Baseline models, this method trains everything from scratch. Then, uses the trained checkpoints to make predictions and then analyzes them to reproduce results of the publications including all main and supplementary figures. 

## Acknowledgements <a name="acknw"></a>

We thank the authors of following open-source repositories. 

- Majority of the functionality in this codebase has been adapted from the chemprop library. 
[Chemprop](http://github.com/chemprop/)
- The rotary positional embeddings functionality
[Rotary PyTorch](https://github.com/lucidrains/rotary-embedding-torch)
- Progres - Protein Graph Embedding Search using pre-trained EGNN models
[Progres](https://github.com/greener-group/progres.git)

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
