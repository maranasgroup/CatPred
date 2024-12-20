# CatPred: A comprehensive framework for deep learning in vitro enzyme kinetic parameters kcat, Km and Ki

[![DOI](https://img.shields.io/badge/DOI-10.1101/2024.03.10.584340-blue)](https://www.biorxiv.org/content/10.1101/2024.03.10.584340v2)
[![Colab](https://img.shields.io/badge/GoogleColab-tiny.cc/catpred-red)](https://tiny.cc/catpred)

## Table of Contents

- [Reproducibility](#reproduce)
- [Google Colab Interface Demo](#web-interface)
- [Local Demo](#local-demo)
- [Acknowledgements](#acknw)
- [License](#license)

## Reproducing publication results <a name="reproduce"></a>

We provide three separate ways for reproducing the results of the publication. 

### 1. Quick method: 

Estimated run time: Few minutes

This is the default script triggered by 'Reproducible Run' 
Can be run using ./code/reproduce_quick.sh

For all results pertaining to CatPred, UniKP, DLKcat and Baseline models, this method only uses pre-trained predictions and analyses to reproduce results of the publications including all main and supplementary figures. 

### 2. Prediction method: 

Estimated run time: Upto a day

Can be run using ./code/reproduce_prediction.sh

Highly recommend running this using the instructions on our Github on a local machine. 

For results pertaining to CatPred, this method uses pre-trained models to perform predictions on test sets. 
For results pertaining to UniKP, DLKcat and Baseline, this method uses only uses pre-trained predictions and analyses to reproduce results of the publications including all main and supplementary figures. 

### 3. Training method: 

Estimated run time: Upto 12-14 days

Can be run using ./code/reproduce_training.sh

Highly recommend running this using the instructions on our Github on a local machine. 

For all results pertaining to CatPred, UniKP, DLKcat and Baseline models, this method trains everything from scratch. Then, uses the trained checkpoints to make predictions and then analyzes them to reproduce results of the publications including all main and supplementary figures. 

Note:- Codes for reproducing Supp. Figures S4-S6 could not be uploaded on CodeOcean because of storage limitations. These will be made available on the Github repository separately. 

## Google Colab Interface Demo (easy) <a name="web-interface"></a>

For ease of use without any hardware requirements, a Google Colab interface is available here: [tiny.cc/catpred](http://tiny.cc/catpred).
It contains sample data, instructions and installation all in the Colab notebook.

## Local Demo <a name="local-demo"></a>

If you would like to install the package on a local machine, please follow the instructions on our Github repository. 
[CatPred](http://github.com/maranasgroup/catpred)

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
