import os
# os.environ['CUDA_VISIBLE_DEVICES']='0, 1, 2, 3'
import argparse
# from random import shuffle
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from train_model import train_model
from sklearn.model_selection import train_test_split
from factor_model import FactorModel
import joblib

    
class CustomDataset(Dataset):
    def __init__(self, data):
        super(CustomDataset, self).__init__()
        self.treatments = data['treatments']
        self.outcomes = data['outcomes']
        self.covariates = data['covariates']
    def __getitem__(self, index):
        treatments = self.treatments[index, :, :]
        previous_treatments = treatments[:-1, :]
        current_treatments = treatments[:, :]
        outcomes = self.outcomes[index, :, :]
        covariates = self.covariates[index, :, :]
        previous_covariates = covariates[:-1, :]
        current_covariates = covariates[:, :]
        return previous_covariates, previous_treatments, current_covariates, current_treatments, outcomes
    def __len__(self):
        return len(self.treatments)



def rmse(y_pred, y_true):
    diff = np.substract(y_pred, y_true)
    square=np.square(diff)
    MSE = square.mean()
    RMSE = np.sqrt(MSE)
    return RMSE

def mape(y_pred, y_true):
    MAPE = np.mean(np.abs((y_true - y_pred)/y_true))*100
    return MAPE

def mae(y_pred, y_true):
    MAE = np.mean(np.abs(y_true - y_pred))
    return MAE
def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_hidden_confounders", default=10, type=int)
    parser.add_argument("--num_ivs", default=10, type=int)
    parser.add_argument("--results_dir", default='results_new')
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--sequence_length", default=16, type=int)
    parser.add_argument('--save_result_with_predict_variables', default=True, type=bool)
    parser.add_argument("--train_and_get_variables", default=True, type=bool)
    parser.add_argument("--alpha", default=0.5, type=float)
    parser.add_argument("--beta", default=0.5, type=float)
    parser.add_argument('--random_seed', type=int, default=40, metavar='G',
                        help='random seed')
    parser.add_argument('--cuda_id', type=int, default=2, metavar='G',
                        help='GPU ID')
    parser.add_argument('--confounding_degree', type=float, default=0.9, metavar='G',
                        help='confounding degree for synthetic data')  
    parser.add_argument('--iv_degree', type=float, default=0.5, metavar='G',
                        help='iv degree for synthetic data')     
    return parser.parse_args()



if __name__ == '__main__':
    args = init_arg()
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    treatment_size = 1
    outcome_size = 1

    print('load data folder', '../data/simulate_{}_{}'.format(args.confounding_degree, args.iv_degree))
    train_dict = joblib.load('../data/simulate_{}_{}/train_simulate_data_16.pkl'.format(args.confounding_degree, args.iv_degree))
    train_data = CustomDataset(train_dict)

    params = {'num_treatments': 1,
              'num_covariates': train_dict['covariates'].shape[-1],
              'num_raw_covariates': train_dict['covariates'].shape[-1],
              'num_confounders': args.num_hidden_confounders,
              'num_ivs': args.num_ivs,
              'num_outcomes': 1,
              'max_sequence_length': args.sequence_length - 1 ,
              'num_epochs': 100}

    print('params[max_sequence_length]', params['max_sequence_length'])

    best_hyperparams = {
        'rnn_hidden_units': 128,
        'fc_hidden_units': 128,
        'learning_rate': 0.001,
        'batch_size': 64, # divide 16 for true batch size
        'lstm_hidden_units': 16
        }

    val_dict = joblib.load('../data/simulate_{}_{}/val_simulate_data_16.pkl'.format(args.confounding_degree, args.iv_degree))
    val_data= CustomDataset(val_dict)

    test_dict = joblib.load('../data/simulate_{}_{}/test_simulate_data_16.pkl'.format(args.confounding_degree, args.iv_degree))
    test_data = CustomDataset(test_dict)
    
    all_dict = dict()
    for key in train_dict.keys():
        all_dict[key] = np.concatenate([train_dict[key], val_dict[key], test_dict[key]], axis=0)
    
    print('all_dict[treatments].shape', all_dict['treatments'].shape)


    all_data = CustomDataset(all_dict)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    all_loader = DataLoader(all_data, batch_size=args.batch_size, shuffle=False)


    factor_model = FactorModel(params, best_hyperparams, args).cuda(args.cuda_id)
    #factor_model = torch.nn.DataParallel(factor_model).cuda(args.cuda_id)
    train_model(factor_model, train_loader, val_loader, args)
    predict_hidden_confounders, predict_ivs = factor_model.compute_hidden_confounders_and_ivs(all_loader)
    predict_hidden_confounders, predict_ivs = predict_hidden_confounders.reshape(-1, train_dict['covariates'].shape[1], args.num_hidden_confounders), predict_ivs.reshape(-1, train_dict['covariates'].shape[1], args.num_ivs)
    print('predict_hidden_confounders.shape', predict_hidden_confounders.shape)
    print('predict_ivs.shape', predict_ivs.shape)

    all_dict['generate_hidden_confounders'] = predict_hidden_confounders
    all_dict['generate_ivs'] = predict_ivs
    
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    if args.save_result_with_predict_variables:
        dataset_new_filename = 'all_simulate_data_with_predicted_variables_{}_{}_{}_{}_new.pkl'.format(params['max_sequence_length'], args.random_seed, args.confounding_degree, args.iv_degree)              
        joblib.dump(all_dict, os.path.join(args.results_dir, dataset_new_filename))
    
        
