# CatPred: Machine Learning models for in vitro enzyme kinetic parameter prediction

***Work in progress:*** Current repository only contains codes and models for prediction. Full training/evaluation codes along with datasets will be released here upon publication.

CatPred predicts in vitro enzyme kinetic parameters (kcat, Km and Ki) using EC, Organism and Substrate features. 

<details open><summary><b>Table of contents</b></summary>


- [Installing pre-requisites](#installation)
- [Usage](#usage)
  - [Input preparation](#preparation)
  - [Making predictions](#prediction)

- [Citations](#citations)
- [License](#license)
</details>

## Installing pre-requisites <a name="installation"></a>

```bash
git clone https://github.com/maranasgroup/catpred.git  # this repo main branch
pip install pandas numpy tqdm
pip install rdkit-pypi
pip install sklearn skops
pip install ete3
```

## Usage <a name="usage"></a>

### Input preparation <a name="preparation"></a>

Prepare an input.csv file as shown in examples/demo.csv 
Best way is to edit the demo.csv file

1. The first column should contain the EC number as per [Enzyme Classification](https://iubmb.qmul.ac.uk/enzyme/). In case of unknown EC enter '-' as place holder. For example, 1.1.1.-

2. The second column should contain the Organism name as per [NCBI Taxonomy](https://www.ncbi.nlm.nih.gov/taxonomy). Common names or short forms will not be processed.

3. The third column should contain a SMILES string. It should be read-able by rdkit [RDKit](https://www.rdkit.org/). 

### Making predictions <a name="prediction"></a>

Use the python script (`python run-catpred.py`):
```
usage: python run-catpred.py [-i] -input INPUT_CSV [-p] -parameter [PARAMETER]

```

The command will first featurize the input file using pre-defined EC and Taxonomy vocabulary. Then, it will add the rdkit fingerprints for SMILES and output the featurized inputs as a pandas dataframe input_feats.pkl. The predictions will be printed and as well as written to output.csv. 

## License <a name="license"></a>

This source code is licensed under the MIT license found in the `LICENSE` file
in the root directory of this source tree.

## Citations <a name="citations"></a>

If you find the models useful in your research, we ask that you cite the relevant paper:

```bibtex
@article{In-preparation,
  author={Boorla, Veda Sheersh and Maranas, Costas D},
  title={CatPred: Machine Learning models for in vitro enzyme kinetic parameter prediction},
  year={2023},
  doi={},
  url={},
  journal={}
}
```
