# Density-Boosts-Robustness-Code
This is the implementation for paper **Density Boosts Everything: A One-stop Strategy for Improving Performance, Robustness, and Sustainability of Malware Detectors**, which is to be appear on NDSS 2025.

## Environments
The code repository is build on a Ubuntu 20.04 with python 3.8.10. 
All required packages are listed in `requirements.txt`; you can use `pip install -r requirements` to install them.

## Datasets
Before starting the reproduction, please download the necessary datasets: [EMBER2017](https://github.com/elastic/ember/tree/d97a0b523de02f3fe5ea6089d080abacab6ee931), [SOREL](https://github.com/sophos/SOREL-20M), [DREBIN2019](https://github.com/s2labres/transcendent-release) and [Contagio](https://contagiodump.blogspot.com/2013/03/16800-clean-and-11960-malicious-files.html). Once you downloaded them, you can denote the path to the datasets in `core/constants.py`.

## Quick stark (on EMBER and SOREL)
Below is the usage of some code files.
1. Prepare the processed datasets by running
```
python process_data.py #dataset for training SCNN
python process_data_histogram.py # dataset for training HistogramNN
python process_data_bundle.py # dataset for training SCBNN
```
2. Train base models (vanillann, lightgbm, ltnn, binarizednn, scnn, scbnn, scbdbnn) for later evaluations, please run `python train_models.py`. Besides, you can train SCBNN or SCNN with different parameters as listed in this file for other evaluations. 
3. Evaluate the backdoor effect by running `python train_backdoored_model.py`, and all results will be saved in `materials/`.




## Description of Files
`core/` is the directory of main codes. 
`materials/` contains all saved files and results. 
`datasets/` is the directory of datasets. 
`modified/` contains the modified code for using original PAD and MAB; you can put these code files to the original PAD/MAB datasets for replicate the experiments.

      
