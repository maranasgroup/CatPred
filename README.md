# CatPred: A Comprehensive Framework for Deep Learning In Vitro Enzyme Kinetic Parameters

[![DOI](https://img.shields.io/badge/DOI-10.1101/2024.03.10.584340-blue)](https://www.nature.com/articles/s41467-025-57215-9)
[![Colab](https://img.shields.io/badge/GoogleColab-tiny.cc/catpred-red)](https://tiny.cc/catpred)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🚨 Announcements 📢

- ✅ **28th Feb 2025** - Published in [_Nature Communications_](https://www.nature.com/articles/s41467-025-57215-9)
- ✅ **27th Dec 2024** - Updated repository with scripts to reproduce results from the manuscript.
- 🚧 **TODO**
  - Add prediction codes for models using 3D-structural features.
  - Add instructions to install CatPred using a Docker image.

---

## 📚 Table of Contents

- [Google Colab Interface](#colab-interface)
- [Local Installation](#local-installation)
   - [System Requirements](#requirements)
   - [Installation](#installing)
   - [Prediction](#predict)
   - [Web API (Optional)](#web-api-optional)
   - [Vercel Deployment (Optional)](#vercel-deployment-optional)
   - [Reproducibility](#reproduce)
- [Acknowledgements](#acknw)
- [License](#license)
- [Citations](#citations)

---

## 🌐 Google Colab Interface <a name="colab-interface"></a>

For ease of use without any hardware requirements, a Google Colab interface is available here: [tiny.cc/catpred](http://tiny.cc/catpred). 
It contains sample data, instructions, and installation all in the Colab notebook.

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
wget -c --tries=5 --timeout=30 https://catpred.s3.us-east-1.amazonaws.com/capsule_data_update.tar.gz || \
wget -c --tries=5 --timeout=30 https://catpred.s3.amazonaws.com/capsule_data_update.tar.gz
tar -xzf capsule_data_update.tar.gz
git clone https://github.com/maranasgroup/catpred.git
cd catpred
conda env create -f environment.yml
conda activate catpred
pip install -e .
````

`stride` is Linux-only and optional for the default demos. If needed for your workflow, install it separately on Linux:

```bash
conda install -c kimlab stride
```

### 🔮 Prediction <a name="predict"></a>

The Jupyter Notebook `batch_demo.ipynb` and the Python script `demo_run.py` show the usage of pre-trained models for prediction.

Input CSV requirements for `demo_run.py` and batch prediction:
- Required columns: `SMILES`, `sequence`, `pdbpath`.
- `pdbpath` must be unique per unique sequence. Reusing the same `pdbpath` for different sequences can produce incorrect cached embeddings.
- Reusing the same `pdbpath` for repeated measurements of the same sequence is supported.

The helper script used to build protein records is:

```bash
python ./scripts/create_pdbrecords.py --data_file <input.csv> --out_file <input.json.gz>
```

CatPred currently expects one sequence per row. Multi-protein complexes (e.g., heteromers/homodimers) are not explicitly modeled as separate chains in the default prediction workflow.

For released benchmark datasets, the number of entries with 3D structure can be smaller than the total sequence/substrate pairs; 3D-derived artifacts are available only for the subset with valid structure mapping.

### 🌍 Web API (Optional)

CatPred also provides an optional FastAPI service for prediction workflows.

Install web dependencies:

```bash
pip install -e ".[web]"
```

Run the API:

```bash
catpred_web --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health` — liveness check.
- `GET /ready` — backend configuration/readiness.
- `POST /predict` — run inference.

By default, the API is hardened for service use:
- `input_file` requests are disabled (use `input_rows` instead).
- request-time overrides of `repo_root` / `python_executable` are disabled.
- `results_dir` is constrained under `CATPRED_API_RESULTS_ROOT`.
- for local backend (and modal requests with fallback enabled), `checkpoint_dir` must resolve under `CATPRED_API_CHECKPOINT_ROOT`.

Minimal `POST /predict` example for local inference using `input_rows`:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "parameter": "kcat",
    "checkpoint_dir": "kcat",
    "input_rows": [
      {"SMILES": "CCO", "sequence": "ACDEFGHIK", "pdbpath": "seq_a"},
      {"SMILES": "CCN", "sequence": "LMNPQRSTV", "pdbpath": "seq_b"}
    ],
    "results_dir": "batch1",
    "backend": "local"
  }'
```

You can keep local inference as default and optionally enable Modal as another backend:

```bash
export CATPRED_DEFAULT_BACKEND=local
export CATPRED_MODAL_ENDPOINT="https://<your-modal-endpoint>"
export CATPRED_MODAL_TOKEN="<optional-token>"
export CATPRED_MODAL_FALLBACK_TO_LOCAL=1
```

Use `"backend": "modal"` in `/predict` requests to route through Modal. If fallback is enabled (env var above or request field `fallback_to_local`), failed modal requests can automatically reroute to local inference.
For local backend requests, place local checkpoints under `CATPRED_API_CHECKPOINT_ROOT` and pass a path relative to that root (for example, `"checkpoint_dir": "kcat"`).

Optional API environment variables:

```bash
# Root directories used by API path constraints
export CATPRED_API_INPUT_ROOT="/absolute/path/for/input-csvs"
export CATPRED_API_RESULTS_ROOT="/absolute/path/for/results"
export CATPRED_API_CHECKPOINT_ROOT="/absolute/path/for/checkpoints"

# Enable only for trusted local workflows (not recommended for public deployments)
export CATPRED_API_ALLOW_INPUT_FILE=1
export CATPRED_API_ALLOW_UNSAFE_OVERRIDES=1

# Request limits
export CATPRED_API_MAX_INPUT_ROWS=1000
export CATPRED_API_MAX_INPUT_FILE_BYTES=5000000
```

Deserialization hardening controls:

```bash
# Trusted roots used by secure loaders (colon-separated list on Unix)
export CATPRED_TRUSTED_DESERIALIZATION_ROOTS="/srv/catpred:/srv/catpred-data"

# Backward-compatible default is enabled (1). Set to 0 to block unsafe pickle-based loading.
# Use 0 only after validating your artifacts are safe-load compatible.
export CATPRED_ALLOW_UNSAFE_DESERIALIZATION=1
```

### ▲ Vercel Deployment (Optional) <a name="vercel-deployment-optional"></a>

This repository includes a Vercel-ready ASGI entrypoint at `api/index.py` and a `vercel.json` route config.

1. Push this repository to GitHub.
2. In Vercel, create a new project from that repo.
3. Set Environment Variables in Vercel Project Settings:

```bash
# Use remote inference backend in serverless deployments
CATPRED_DEFAULT_BACKEND=modal
CATPRED_MODAL_ENDPOINT=https://<your-modal-endpoint>
CATPRED_MODAL_TOKEN=<optional-token>
CATPRED_MODAL_FALLBACK_TO_LOCAL=0
```

Notes:
- Serverless filesystems are ephemeral/read-only except `/tmp`; this app auto-uses `/tmp/catpred` on Vercel.
- Local checkpoint-based inference is not recommended on Vercel serverless due runtime/dependency limits.
- If `CATPRED_MODAL_ENDPOINT` is not configured, the UI still loads but prediction requests will be limited by backend readiness.

#### Deploy a Modal endpoint for Vercel

This repo includes `modal_app.py`, a Modal `POST` endpoint compatible with CatPred's `/predict` modal backend contract.

1. Install and authenticate Modal CLI:

```bash
pip install modal
modal setup
```

2. Create/upload checkpoints into a Modal Volume (one-time):

```bash
modal volume create catpred-checkpoints
modal volume put catpred-checkpoints ./checkpoints/kcat kcat
modal volume put catpred-checkpoints ./checkpoints/km km
modal volume put catpred-checkpoints ./checkpoints/ki ki
```

3. (Recommended) create a secret token for endpoint auth:

```bash
modal secret create catpred-modal-auth CATPRED_MODAL_AUTH_TOKEN="<your-token>"
```

4. Deploy:

```bash
modal deploy modal_app.py
```

After deploy, copy the printed endpoint URL (for function `predict`) and set Vercel variables:

```bash
CATPRED_DEFAULT_BACKEND=modal
CATPRED_MODAL_ENDPOINT=https://<your-modal-endpoint>
CATPRED_MODAL_TOKEN=<your-token>
CATPRED_MODAL_FALLBACK_TO_LOCAL=0
```

#### CI/CD (GitHub Actions + Vercel + Modal)

This repo includes two GitHub Actions workflows:

- `.github/workflows/ci.yml`
  - Runs on every PR and push to `main`.
  - Installs minimal API dependencies, compiles all Python files, and smoke-tests API entrypoints.
- `.github/workflows/deploy-modal.yml`
  - Runs on push to `main` when backend files change (and manually via `workflow_dispatch`).
  - Deploys `modal_app.py` automatically.

To enable automatic Modal deploys from GitHub Actions, add repository secrets:

- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`

Create these from Modal:

1. Go to [https://modal.com/settings/tokens](https://modal.com/settings/tokens).
2. Create a token with deploy permissions for your workspace.
3. Copy token ID and secret into GitHub repo settings:
   `Settings -> Secrets and variables -> Actions -> New repository secret`.

Vercel deployment remains automatic from the connected GitHub branch (`main`).

### 🧪 Fine-Tuning On Custom Data

You can fine-tune CatPred on your own regression targets using `train.py`.

1. Prepare train/val/test CSVs with at least:
- `SMILES`
- `sequence`
- `pdbpath` (unique per unique sequence)
- one numeric target column (for example: `log10kcat_max`)

2. Build a protein-records file that covers all `pdbpath` values in your splits:

```bash
python ./scripts/create_pdbrecords.py --data_file <combined_or_train_csv> --out_file <protein_records.json.gz>
```

3. Train:

```bash
python train.py \
  --protein_records_path <protein_records.json.gz> \
  --data_path <train.csv> \
  --separate_val_path <val.csv> \
  --separate_test_path <test.csv> \
  --dataset_type regression \
  --smiles_columns SMILES \
  --target_columns <target_column_name> \
  --add_esm_feats \
  --loss_function mve \
  --save_dir <output_checkpoint_dir>
```

For working end-to-end examples, see the training commands in scripts such as `scripts/reproduce_figS10_catpred.sh`.

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
