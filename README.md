## Generative machine learning for multidisease prevention and early detection from integrated genetic, lifestyle and clinical data

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-blue)](https://creativecommons.org/licenses/by/4.0/)&nbsp;&nbsp;
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB.svg?logo=Python)](https://www.python.org/downloads/)&nbsp;&nbsp;
[![PyTorch 2.8](https://img.shields.io/badge/torch-2.8-ee4c2c.svg?logo=Pytorch&logoColor=%23EE4C2C)](https://pytorch.org/)

**Author**: María Alonso-Vega González

**Supervisors**:
- Diego Quintana Torres
- Romina Astrid Rebrij

## Repository Overview

This repository contains the code for the GPT-2-based model implemented in the final master's thesis "Generative machine learning for multidisease prevention and early detection from integrated genetic, lifestyle and clinical data", along with the training code and analysis notebooks.

The implementation is based on the [Delphi](https://github.com/gerstung-lab/Delphi) model.

## Installation

1. Download the repository:

```bash
git clone https://github.com/mariavg2409/TFM-2026-MV.git
cd Delphi
```

2. Create a virtual conda environment and install the requirements:
```bash
conda create -n tfm_2026_mv python=3.11
conda activate tfm_2026_mv
pip install -r requirements.txt
```

## Training

To train the model, run:

```bash
python train.py config/train_delphi_demo.py --device=cuda
```

If you want to train the model on a CPU, remove the `--device=cuda` argument.
For more information on all the available training options, check the `config/train_delphi_demo.py` file.

> [!NOTE]
> We recommend using `wandb` to track the training runs (`--wandb_log=True`). You can create an account and login by following the instructions [here](https://wandb.ai/site/).

## Notebooks

To reproduce the results in this project, please refer to the Jupyter notebooks in the `notebooks/` directory. This directory contains the following notebooks:

- `1_filter_ukb_data.ipynb`: Script used to filter the ICD-10 coded diseases in order to obtain the same ones used in Delphi for training.

- `2_embeddings_creation.ipynb`: Discretization of the BMI variable and embedding generation for both exposome variables and diseases.

- `3_training_analysis.ipynb`: Training plots, please be sure to repect the order of the loading data in order to mantain the order of the plots and its headers/colours.

- `4_roc_curves.ipynb`: Plots of the ROC curves and AUC calculation for 2 random diseases, take into account that the number of diseases to plot can be chosen by the user



## Citation

```bibtex
@mastersthesis{mariavg2026,
  author = {María Alonso Vega},
  title = {Generative machine learning for multidisease prevention and early detection from integrated genetic, lifestyle and clinical data},
  school = {Universitat Oberta de Catalunya and Universitat de Barcelona},
  year = {2026},
  type = {Trabajo Fin de Máster},
  note = {Supervisors: Diego Quintana Torres and Romina Astrid Rebrij}
}
```
