# Density-Boosts-Robustness-Code
This is the implementation for paper **Density Boosts Everything: A One-stop Strategy for Improving Performance, Robustness, and Sustainability of Malware Detectors**, which is published on NDSS 2025. 

## Directly-related Code
If you only care about the core algorithm of this paper, see `process_data_bundle.py` for implementation of _subspace compression_ with bundling and function `def train()` in `core/utils.py` for implementation of _density boosting_.

## Environments
The code repository is build on a Ubuntu 20.04 with python 3.8.10.  
All required packages are listed in `requirements.txt`; you can use `pip install -r requirements` to install them.

## Datasets
Before starting the reproduction, please download the necessary datasets: [EMBER2017](https://github.com/elastic/ember/tree/d97a0b523de02f3fe5ea6089d080abacab6ee931), [SOREL](https://github.com/sophos/SOREL-20M), [DREBIN2019](https://github.com/s2labres/transcendent-release) and [Contagio](https://contagiodump.blogspot.com/2013/03/16800-clean-and-11960-malicious-files.html). Once you downloaded them, you can denote the path to the datasets in `core/constants.py`.

## Quick stark (on EMBER and SOREL)
Below is the usage of code files.
1. Prepare the processed datasets by running
```
python process_data.py #dataset for training SCNN
python process_data_histogram.py # dataset for training His togramNN
python process_data_bundle.py # dataset for training SCBNN
```
2. Train base models (vanillann, lightgbm, ltnn, binarizednn, scnn, scbnn, scbdbnn) for later evaluations, please run `python train_models.py`. Besides, you can train SCBNN or SCNN with different parameters as listed in this file for other evaluations.
4. Evaluate the concept drift on Sorel-20M by running `python evaluating_concept_drift_on_SOREL.py`, and all results will be saved in `materials/evaluation_on_sorel.txt`
5. Evaluate the backdoor effect by running `python train_backdoored_model.py`, and all results will be saved in `materials/`.
6. Before start evaluating evasion attacks, run `python app.py` to wrap all models into apis. Then refer to `gamma_attack.py` for evaluating GAMMA Evasion on different models and refer to [MAB-malware](https://github.com/weisong-ucr/MAB-malware) to evaluating MAB evasion (you can put files in `modified_MAB/`  to MAB docker to implement attacks against our wrapped apis.).

### Training of PAD model
Once you prepared the dataset (clean or poisoned), you can replace [original PAD files](https://github.com/deqangss/pad4amd) with our modified code(see below) in `modified_pad/` and run
```sh
$ python -m examples.amd_pad_ma_test --cuda --use_cont_pertb --beta_1 0.1 --beta_2 1.0 --lambda_lb 1.0 --lambda_ub 1.0 --seed 0 --batch_size 128 --proc_number 10 --epochs 50 --max_vocab_size 10000 --dense_hidden_units "1024,512,256" --weight_decay 0.0 --lr 0.001 --dropout 0.6  --ma "stepwise_max" --steps_l1 50 --steps_linf 50 --step_length_linf 0.02 --steps_l2 50 --step_length_l2 0.5 --is_score_round
```
to launch the training and use the model to do the later verification. The key problem is to transform the 

## Evaluation on DREBIN (Android)
Here descripte the steps of evaluation on DREBIN datasets. 
1. Similarly, prepare the processed datasets by running
```sh
$ python process_data_drebin.py # dataset for training NN-Bundle (DREBIN)
```
2. Training the model and evaluate the performance and mimicry attacks by using `Experiments-DREBIN.ipynb` which includes most DREBIN-related experiments.
3. After you train the base models, you can refer to `train_backdoored_drebin.py` for backdoor evaluation on DREBIN.
4. Also, please base on original codes to train the PAD model, you can refer to our files in `modified_PAD/` (mainly see `base_attack.py` and `amd_pad_map.py`) for modification. After that, run 
```sh
$ python -m examples.amd_pad_ma_test --cuda --use_cont_pertb --beta_1 0.1 --beta_2 1.0 --lambda_lb 1.0 --lambda_ub 1.0 --seed 0 --batch_size 128 --proc_number 10 --epochs 50 --max_vocab_size 10000 --dense_hidden_units "1024,512,256" --weight_decay 0.0 --lr 0.001 --dropout 0.6  --ma "stepwise_max" --steps_l1 50 --steps_linf 50 --step_length_linf 0.02 --steps_l2 50 --step_length_l2 0.5 --is_score_round
```

## Evaluation on Contagio (PDF)
Here descripte the steps of evaluation on Contagio datasets. 
1. Prepare the processed datasets by running
```sh
$ python process_data.py # Set the dataset as Contagio first (Extracted as dumped numpy)
$ python process_data_histogram.py # Please set the dataset as Contagio first(Extracted as dumped numpy)
$ python process_data_bundle.py # Please set the dataset as Contagio first (Extracted as dumped numpy)
```
2. Training the model and evaluate the performance and mimicry attacks by using `Experiments-pdf.ipynb` which includes most PDF-related experiments.
3. After you train the base models, you can refer to `train_backdoored_pdf.py` for backdoor evaluation on DREBIN.
4. Again, please refer original codes to train the PAD model, you can refer to our files in `modified_PAD/` (mainly see `base_attack.py` and `amd_pad_map.py`) for modification. After that, run 
```sh
$ python -m examples.amd_pad_ma_test --cuda --use_cont_pertb --beta_1 0.1 --beta_2 1.0 --lambda_lb 1.0 --lambda_ub 1.0 --seed 0 --batch_size 128 --proc_number 10 --epochs 50 --max_vocab_size 10000 --dense_hidden_units "1024,512,256" --weight_decay 0.0 --lr 0.001 --dropout 0.6  --ma "stepwise_max" --steps_l1 50 --steps_linf 50 --step_length_linf 0.02 --steps_l2 50 --step_length_l2 0.5 --is_score_round
```

## Description of Files
```
|+---app.py: This file help wrap all models into api for remote access (e.g. from docker).
|+---core/ is the directory of main codes.  
|    |     model_utils.py: This module is for model usages, e.g. loading, training, saving, evaluating models, explanation (SHAP).  
|    |     data_utils.py: This module is for processing dataset, e.g. loading/saving dataset, loading/saving/processing features.   
|    |     constants.py: This module is used to set some constant values, like configuration.  
|    |     feature_selectors.py: This module is used to implement backdoor attacks.  
|    |     models.py: This module contains code for training/loading specific models (for easy usages).  
|    |     nn.py;mmsnn.py;ltnn.py;malconv.py`: Definition for different models.  
|    |     utils.py: This module is for different tools/algorithms/implementations. 
|+---materials/ contains all saved files and results.  
|+---datasets/ is the directory of datasets. 
|+---pad/ is the directory of PAD code, we modified it for simplifying usage of the API. Please refer to [its original version](https://github.com/deqangss/pad4amd) for training models. 
|+---modified_MAB/ contains the modified code for MAB; you can put these code files to the original MAB directory.
|    |+---chart/: we modified some figure drawing scripts.
|    classifier.py: add a remote classifier.
|    malwares.txt: malwares used for evaluation.
|    models.py: add the definition of remote(out of docker) classifier.
|    samples_manager.py: randomly select samples.
|    test_samples.py: replay evasive samples.
|+---modified_PAD/ contains the modified code for PAD; you can put these code files to the original PAD directory.
|    |    base_attack.py: We only changed its manipultable features for ember and PDF (Having some additional-only features or unmodifiable features is critical for the defensive effect of PAD models, the PAD can build a strong convex outer bound with these features. ).  Use this file to replace `pad4amd/core/attack/base_attack.py`.  
|    |    amd_icnn.py: When we ran the PAD code, we met an small problem; replace `core/defense/amd_icnn.py` with this file can eliminate the error.  
|    |    amd_pad_ma.py: We added our density boosting strategy in this file; if you wanna train the model with SCBNN-DB-PAD, you might need this file. Refer to "class AMalwareDetectionPAD_density" for the modification and replace `pad4amd/core/defense/amd_pad_ma.py` with this file to use it.  
|    |    amd_pad_ma_test.py: This is the script of training PAD models, you may need to have your own modification, such as replacing the dataset with clean or poisoned dataset and training strategies(with or without density boosting). Replace `pad4amd/example/amd_pad_ma_test.py` with this file. 
```


## Contact
If you encounter any problems when using the code in this repository, please feel free to contact me (jianwentian1994@foxmail.com) or raise an issue. 







      
