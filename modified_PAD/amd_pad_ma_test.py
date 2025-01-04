from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path as path
import time
from functools import partial
from torch.utils.data import TensorDataset, DataLoader
import sys
sys.path.append("core")
sys.path.append("./")
sys.path.append("./tools")
from core.defense import Dataset
from core.defense import MalwareDetectionDNN, AdvMalwareDetectorICNN, AMalwareDetectionPAD, AMalwareDetectionPAD_density
from core.attack import Max, PGD, PGDl1, StepwiseMax
from tools.utils import save_args, get_group_args, dump_pickle
from examples.amd_icnn_test import cmd_md
import numpy as np
import ember
import joblib
import torch

max_adv_argparse = cmd_md.add_argument_group(title='max adv training')
max_adv_argparse.add_argument('--beta_1', type=float, default=0.1, help='penalty factor on adversarial loss.')
max_adv_argparse.add_argument('--beta_2', type=float, default=1., help='penalty factor on adversary detector.')
max_adv_argparse.add_argument('--lambda_lb', type=float, default=1e-3,
                              help='the lower bound of penalty factor on adversary detector for looking for attacks.')
max_adv_argparse.add_argument('--lambda_ub', type=float, default=1e3,
                              help='the upper bound of penalty factor on adversary detector for looking for attacks.')
max_adv_argparse.add_argument('--detector', type=str, default='icnn',
                              choices=['none', 'icnn'],
                              help="detector type, either of 'icnn' and 'none'.")
max_adv_argparse.add_argument('--under_sampling', type=float, default=1.,
                              help='under-sampling ratio for adversarial training')
max_adv_argparse.add_argument('--ma', type=str, default='max', choices=['max', 'stepwise_max'],
                              help="Type of mixture of attack: 'max' or 'stepwise_max' strategy.")
max_adv_argparse.add_argument('--steps_l1', type=int, default=50,
                              help='maximum number of perturbations.')
max_adv_argparse.add_argument('--steps_l2', type=int, default=50,
                              help='maximum number of steps for base attacks.')
max_adv_argparse.add_argument('--step_length_l2', type=float, default=0.5,
                              help='step length in each step.')
max_adv_argparse.add_argument('--steps_linf', type=int, default=50,
                              help='maximum number of steps for base attacks.')
max_adv_argparse.add_argument('--step_length_linf', type=float, default=0.02,
                              help='step length in each step.')
max_adv_argparse.add_argument('--random_start', action='store_true', default=False,
                              help='randomly initialize the start points.')
max_adv_argparse.add_argument('--round_threshold', type=float, default=0.5,
                              help='threshold for rounding real scalars at the initialization step.')
max_adv_argparse.add_argument('--is_score_round', action='store_true', default=False,
                              help='whether scoring rule takes as input with rounding operation or not.')
max_adv_argparse.add_argument('--use_cont_pertb', action='store_true', default=False,
                              help='whether use the continuous perturbations for adversarial training.')


def gini(C):
    #l1-norm
    l1 = np.abs(C).sum()
    N = len(C)
    s = 0
    for k,i in enumerate(sorted(C)):
        s += (i/l1)*((N-(k+1)+0.5)/N)
    gi = 1 - 2*s
    return gi

def get_coredict(x_train):
    value_spaces = []
    coredict = dict()
    coredict['sparsity_list'] = np.zeros(x_train.shape[1])
    coredict["available_indicies"] = np.zeros(x_train.shape[1])
    for i in range(x_train.shape[1]):
        coredict[i] = dict()
        unique_value_counts = bincount(x_train[:,i])#convert into Tensor in case of precision inconsistency
        for v,c in unique_value_counts:
            coredict[i][v] = c
        if len(coredict[i].values())>1:#has more values
            coredict["available_indicies"][i] = 1
    print(f'Available indicies: {sum(coredict["available_indicies"])}')
    return coredict

def bincount(arr):
    if isinstance(arr, torch.Tensor):
        unique_values, index = torch.unique(arr, return_inverse=True)
        unique_values = unique_values.tolist()
        index = index.tolist()
    else:
        unique_values, index = np.unique(arr, return_inverse=True)
    count = np.bincount(index)
    unique_value_counts = list(zip(unique_values, count))
    return unique_value_counts

def transform_as_prob(coredict):
    for idx in range(len(coredict['sparsity_list'])):
        keys, values = list(coredict[idx].keys()), list(coredict[idx].values())
        values = 1/np.array(values)
        for v in keys:
            coredict[idx][v] = (1/coredict[idx][v])/values.sum()#prob
        #print(idx,coredict[idx])

def _main(n):
    args = cmd_md.parse_args()
    tag = 2017
    ratio = 8
    #EMBER dataset
    x_train, x_test, y_train, y_test = joblib.load(f"../materials/compressed_{tag}_{ratio}_js.pkl")
    #x_train,y_train,x_test,y_test = joblib.load(f"../materials/poisoned_x_{n}_data-defense_all_pad.pkl")
    #x_train,y_train,x_test,y_test,s = joblib.load(f"../materials/poisoned_mms.pkl")
    x_val = x_test

    # DREBIN dataset
    #tag = 2015
    #ratio = 558
    #x_train, x_val, x_test, y_train, y_val, y_test = joblib.load(f"../materials/drebin_{tag}_{fn}.pkl")
    #x_train, x_test,y_train,y_test = joblib.load(f"../materials/compressed_drebin_{tag}_{ratio}.pkl")
    #x_train, x_val, x_test, y_train, y_val, y_test = joblib.load(f"../materials/drebin_2015_682.pkl")
    #x_train, y_train, x_val, y_val, x_test, y_test = joblib.load(f"../materials/poisoned_x_500_data-defense_{typ}_{n}_pad.pkl")
    #x_train, y_train, x_val, y_val, x_test, y_test = joblib.load(f"../materials/drebin_selected.pkl")
    #x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.05,random_state = 3) 

    # PDF dataset
    ##x_train, x_val, x_test, y_train, y_val, y_test = joblib.load(f"../pdf_dataset.pkl")

    fn = x_train.shape[1]
    coredict = get_coredict(x_train)
    transform_as_prob(coredict)
    
    train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
    val_dataset = TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val))
    test_dataset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
    batch_size = 512  
    train_dataset_producer = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # shuffle=True 表示在每个 epoch 中打乱数据
    val_dataset_producer = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # shuffle=True 表示在每个 epoch 中打乱数据
    test_dataset_producer = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # shuffle=True 表示在每个 epoch 中打乱数据

    # test: model training
    if not args.cuda:
        dv = 'cpu'
    else:
        dv = 'cuda'

    model_name = args.model_name if args.mode == 'test' else time.strftime("%Y%m%d-%H%M%S")
    model = MalwareDetectionDNN(x_train.shape[1],
                                2,
                                device=dv,
                                name=model_name,
                                **vars(args)
                                )
    if args.detector == 'icnn':
        model = AdvMalwareDetectorICNN(model,
                                       input_size=x_train.shape[1],
                                       n_classes=2,
                                       device=dv,
                                       name=model_name,
                                       **vars(args)
                                       )

    else:
        raise NotImplementedError
    model = model.to(dv).double()
    pgdlinf = PGD(norm='linf', use_random=False,
                  is_attacker=False,
                  device=model.device)
    pgdlinf.perturb = partial(pgdlinf.perturb,
                              steps=args.steps_linf,
                              step_length=args.step_length_linf,
                              verbose=False
                              )
    pgdl2 = PGD(norm='l2', use_random=False, is_attacker=False, device=model.device)
    pgdl2.perturb = partial(pgdl2.perturb,
                            steps=args.steps_l2,
                            step_length=args.step_length_l2,
                            verbose=False
                            )
    pgdl1 = PGDl1(is_attacker=False, device=model.device)
    pgdl1.perturb = partial(pgdl1.perturb,
                            steps=args.steps_l1,
                            verbose=False)

    if args.ma == 'max':
        attack = Max(attack_list=[pgdlinf, pgdl2, pgdl1],
                     varepsilon=1e-9,
                     is_attacker=False,
                     device=model.device
                     )
        attack_param = {
            'steps_max': 1,  # steps for max attack
            'verbose': True
        }
    elif args.ma == 'stepwise_max':
        attack = StepwiseMax(is_attacker=False, device=model.device)
        attack_param = {
            'steps': max(max(args.steps_l1, args.steps_linf), args.steps_l2),
            'sl_l1': 1.,
            'sl_l2': args.step_length_l2,
            'sl_linf': args.step_length_linf,
            'is_score_round': args.is_score_round,
            'verbose': True
        }
    else:
        raise NotImplementedError("Expected 'max' and 'stepwise_max'.")

    #max_adv_training_model = AMalwareDetectionPAD_density(model, attack, attack_param) #use density-boosting
    max_adv_training_model = AMalwareDetectionPAD(model, attack, attack_param) #no density-boosting
    if args.mode == 'train':
        max_adv_training_model.load()
        max_adv_training_model.fit(train_dataset_producer,
                                   val_dataset_producer,
                                   adv_epochs=args.epochs,
                                   beta_1=args.beta_1,
                                   beta_2=args.beta_2,
                                   lmda_lower_bound=args.lambda_lb,
                                   lmda_upper_bound=args.lambda_ub,
                                   use_continuous_pert=args.use_cont_pertb,
                                   lr=args.lr,
                                   under_sampling_ratio=args.under_sampling,
                                   weight_decay=args.weight_decay,
				                   #coredict = False #use density-boosting, set it True
                                   )

        # get threshold
        max_adv_training_model.load()
        max_adv_training_model.model.get_threshold(val_dataset_producer)
        max_adv_training_model.save_to_disk(max_adv_training_model.model_save_path)
        # human readable parameters
        save_args(path.join(path.dirname(max_adv_training_model.model_save_path), "hparam"), vars(args))
        # save parameters for rebuilding the neural nets
        dump_pickle(vars(args), path.join(path.dirname(max_adv_training_model.model_save_path), "hparam.pkl"))
    # test: accuracy
    max_adv_training_model.load()
    max_adv_training_model.model.get_threshold(val_dataset_producer, ratio=args.ratio)
    max_adv_training_model.model.predict(test_dataset_producer, indicator_masking=True)


if __name__ == '__main__':
    _main(n)
