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
pip install pandas numpy tqdm rdkit-pypi scikit-learn==1.3.2 skops==0.9.0 ete3
```
Clone this repo, download the data folder and extract into root directory 
```bash
git clone https://github.com/maranasgroup/catpred.git  # this repo main branch
cd catpred
wget https://catpred.s3.amazonaws.com/data.tar.gz
tar -xvzf data.tar.gz
```

Download pre-trained models and extract into root directory
```bash
wget https://catpred.s3.amazonaws.com/models.tar.gz
tar -xvzf models.tar.gz
```
## Usage <a name="usage"></a>

### Input preparation <a name="preparation"></a>

Prepare an input.csv file as shown in examples/demo.csv 
Best way is to edit the demo.csv file

1. The first column should contain the EC number as per [Enzyme Classification](https://iubmb.qmul.ac.uk/enzyme/). 
In case of unknown EC number at a particular level, use '-' as a place holder. For example, if the last two levels are unknown then, use 1.1.1.-

2. The second column should contain the Organism name as per [NCBI Taxonomy](https://www.ncbi.nlm.nih.gov/taxonomy). 
Common names or short forms will not be processed. In case of a rare Organism or a new strain, use the [NCBI Taxonomy](https://www.ncbi.nlm.nih.gov/taxonomy) website to find the Organism that you think is the closest match.

3. The third column should contain a SMILES string. It should be read-able by rdkit [RDKit](https://www.rdkit.org/). You can use [PubChem](https://pubchem.ncbi.nlm.nih.gov/) or [BRENDA-Ligand](https://www.brenda-enzymes.org/structure_search.php) or [CHE-EBI](https://www.ebi.ac.uk/chebi/) to search for SMILES. Alternatively, you can use [PubChem-Draw](https://pubchem.ncbi.nlm.nih.gov//edit3/index.html) to generate SMILES string for any molecule you draw.

### Making predictions <a name="prediction"></a>

Use the python script (`python run-catpred.py`):
```
usage: python run-catpred.py [-i] -input INPUT_CSV [-p] -parameter [PARAMETER]

```

The command will first featurize the input file using pre-defined EC and Taxonomy vocabulary. Then, it will add the rdkit fingerprints for SMILES and output the featurized inputs as a pandas dataframe input_feats.pkl. 

The predictions will be printed to the the screen and as well as written to a .csv file with a name INPUT_CSV_preds.csv

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
