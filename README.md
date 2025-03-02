# CatPred: A Comprehensive Framework for Deep Learning In Vitro Enzyme Kinetic Parameters kcat, Km, and Ki 🧬

[![DOI](https://img.shields.io/badge/DOI-10.1101/2024.03.10.584340-blue)](https://www.nature.com/articles/s41467-025-57215-9)
[![Colab](https://img.shields.io/badge/GoogleColab-tiny.cc/catpred-red)](https://tiny.cc/catpred)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🚨 Announcements 📢

- ✅ **28th Feb 2025** - Published in _Nature Communications._
- ✅ **27th Dec 2024** - Updated repository with scripts to reproduce results from the manuscript.
- 🚧 **TODO** - Add prediction codes for models using 3D-structural features.

---

## 📚 Table of Contents

- [Google Colab Interface](#colab-interface)
- [Local Installation](#local-installation)
   - [System Requirements](#requirements)
   - [Installation](#installing)
   - [Prediction](#predict)
   - [Reproducibility](#reproduce)
- [Acknowledgements](#acknw)
- [License](#license)
- [Citations](#citations)

---

## 🌐 Google Colab Interface <a name="colab-interface"></a>

For ease of use without any hardware requirements, a Google Colab interface is available here: [tiny.cc/catpred](http://tiny.cc/catpred). It contains sample data, instructions, and installation all in the Colab notebook.

---

## 💻 Local Installation <a name="local-installation"></a>

If you would like to install the package on a local machine, please follow the instructions below.

### 🖥️ System Requirements <a name="requirements"></a>

- **For prediction:** Any machine running a Linux-based operating system is recommended.
- **For training:** A Linux-based operating system on a GPU-enabled machine is recommended.

Both training and prediction have been tested on **Ubuntu 20.04.5 LTS** with **NVIDIA A10** and **CUDA Version: 12.0**.

To train or predict with GPUs, you will need:
- **CUDA >= 11.7**
- **cuDNN**

### 📥 Installation <a name="installing"></a>

Both options require **conda**, so first install Miniconda from [https://conda.io/miniconda.html](https://conda.io/miniconda.html).

Then proceed to either option below to complete the installation. If installing the environment with conda seems to be taking too long, you can also try running `conda install -c conda-forge mamba` and then replacing `conda` with `mamba` in each of the steps below.

**Note for machines with GPUs:** You may need to manually install a GPU-enabled version of PyTorch by following the instructions [here](https://pytorch.org/get-started/locally/). If you're encountering issues with not using a GPU on your system after following the instructions below, check which version of PyTorch you have installed in your environment using `conda list | grep torch` or similar. If the PyTorch line includes `cpu`, please uninstall it using `conda remove pytorch` and reinstall a GPU-enabled version using the instructions at the link above.

#### Installing and Downloading Pre-trained Models (~10 mins)

```bash
mkdir catpred_pipeline catpred_pipeline/results
cd catpred_pipeline
wget https://catpred.s3.us-east-1.amazonaws.com/capsule_data.tar.gz
tar -xzf capsule_data.tar.gz
git clone https://github.com/maranasgroup/catpred.git
cd catpred
conda env create -f environment.yml
conda activate catpred
pip install -e .
````

### 🔮 Prediction <a name="predict"></a>

The Jupyter Notebook `batch_demo.ipynb` and the Python script `demo_run.py` show the usage of pre-trained models for prediction.

### 🔄 Reproducing Publication Results <a name="reproduce"></a>

We provide three separate ways for reproducing the results of the publication.

#### 1. Quick Method ⚡

**Estimated run time:** Few minutes

Run using:
```bash
./reproduce_quick.sh
```

For all results pertaining to CatPred, UniKP, DLKcat, and Baseline models, this method only uses pre-trained predictions and analyses to reproduce results of the publications, including all main and supplementary figures.

#### 2. Prediction Method 🛠️

**Estimated run time:** Up to a day depending on your GPU

Run using:
```bash
./reproduce_prediction.sh
```

For results pertaining to CatPred, this method uses pre-trained models to perform predictions on test sets. For results pertaining to UniKP, DLKcat, and Baseline, this method uses only pre-trained predictions and analyses to reproduce results of the publications, including all main and supplementary figures.

#### 3. Training Method 🏋️

**Estimated run time:** Up to 12-14 days depending on your GPU

Run using:
```bash
./reproduce_training.sh
```

For all results pertaining to CatPred, UniKP, DLKcat, and Baseline models, this method trains everything from scratch. Then, it uses the trained checkpoints to make predictions and analyzes them to reproduce results of the publications, including all main and supplementary figures.

---

## 🙏 Acknowledgements <a name="acknw"></a>

We thank the authors of the following open-source repositories:

- **Chemprop** - Majority of the functionality in this codebase has been inspired from the [Chemprop](http://github.com/chemprop/) library.
- **Rotary PyTorch** - The rotary positional embeddings functionality for Seq-Attn. is from [Rotary PyTorch](https://github.com/lucidrains/rotary-embedding-torch).
- **Progres** - Protein Graph Embedding Search using pre-trained EGNN models from [Progres](https://github.com/greener-group/progres.git).

---

## 📜 License <a name="license"></a>

This source code is licensed under the MIT license found in the `LICENSE` file in the root directory of this source tree.

---

## 📖 Citations <a name="citations"></a>

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
