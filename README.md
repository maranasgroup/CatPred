# CatPred: A comprehensive framework for deep learning in vitro enzyme kinetic parameters kcat, Km and Ki

[![DOI](https://img.shields.io/badge/DOI-10.1101/2024.03.10.584340-blue)](https://www.biorxiv.org/content/10.1101/2024.03.10.584340v2)
[![Colab](https://img.shields.io/badge/GoogleColab-tiny.cc/catpred-red)](https://tiny.cc/catpred)


## Table of Contents

- [Google Colab Interface Demo](#web-interface)
- [Local Demo](#local-demo)
  * [System Requirements](#requirements)
  * [Installing using pip](#installing)
  * [Run demo](#run-demo)
- [Acknowledgements](#acknw)
- [License](#license)

## Google Colab Interface Demo (easy) <a name="web-interface"></a>

For ease of use without any hardware requirements, a Google Colab interface is available here: [tiny.cc/catpred](http://tiny.cc/catpred).
It contains sample data, instructions and installation all in the Colab notebook.

## Local Demo <a name="local-demo"></a>

If you would like to install the package on a local machine, please follow the following instructions.

### System Requirements <a name="requirements"></a>

For using pre-trained models to predict, any machine running a Linux based operating system is recommended.
For training, we recommend using a Linux based operating system on a GPU-enabled machine.

Both training and prediction have been tested on Ubuntu 20.04.5 LTS with NVIDIA A10 and CUDA Version: 12.0

To train with GPUs, you will need:
 * cuda >= 11.7
 * cuDNN

### Installation <a name="installing"></a>

Both options require conda, so first install Miniconda from [https://conda.io/miniconda.html](https://conda.io/miniconda.html).

Then proceed to either option below to complete the installation. If installing the environment with conda seems to be taking too long, you can also try running `conda install -c conda-forge mamba` and then replacing `conda` with `mamba` in each of the steps below.

**Note for machines with GPUs:** You may need to manually install a GPU-enabled version of PyTorch by following the instructions [here](https://pytorch.org/get-started/locally/). If you're encountering issues with not using a GPU on your system after following the instructions below, check which version of PyTorch you have installed in your environment using `conda list | grep torch` or similar. If the PyTorch line includes `cpu`, please uninstall it using `conda remove pytorch` and reinstall a GPU-enabled version using the instructions at the link above.

#### Installing and downloading pre-trained models (~5 mins)

1. `git clone https://github.com/maranasgroup/catpred.git`
2. `cd CatPred`
3. `conda env create -f environment.yml`
4. `conda activate catpred`
5. `pip install -e .`
6. `pip install ipdb fair-esm rotary_embedding_torch==0.6.5 egnn_pytorch -q`
7. `wget https://catpred.s3.amazonaws.com/production_models.tar.gz -q`
8. `wget https://catpred.s3.amazonaws.com/processed_databases.tar.gz -q`
9. `tar -xzf production_models.tar.gz`
10. `tar -xzf processed_databases.tar.gz`

### Run a demo (~2 mins) <a name="run-demo"></a>

Use the `demo.ipynb` jupyter notebook to run the demo. 

## Reproducing publication training/results

To reproduce publication results, download and extract the scripts and required data using
```
wget https://catpred.s3.amazonaws.com/reproduce_publication_results.tar.gz 
mkdir reproduce_publication_results.tar.gz; mv reproduce_publication_results.tar.gz ./reproduce_publication_results
tar -xvzf reproduce_publication_results.tar.gz
```

In order to train publication models, you must download and extract training datasets using
```
wget https://catpred.s3.amazonaws.com/publication_training_datasets.tar.gz
tar -xvzf publication_training_datasets.tar.gz
```

### Training

TODO: Will be made available upon publication
```
```

## Acknowledgements <a name="acknw"></a>

We thank the authors of following open-source repositories. 

- Majority of the functionality in this codebase has been adapted from the chemprop library. 
[Chemprop](http://github.com/chemprop/)
- The rotary positional embeddings functionality
[Rotary PyTorch](https://github.com/lucidrains/rotary-embedding-torch)

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
