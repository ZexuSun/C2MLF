from sklearn.model_selection import train_test_split
import pickle
from torch.utils.data import Dataset, DataLoader
from rmsn_outcome import Propensity_Net_Num, Propensity_Net_Den, train_propensity_net, Encoder, train_encoder, Decoder, train_decoder, test_evaluation
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.distributions import Normal
import joblib

class CustomDataset(Dataset):
    def __init__(self, data, with_hidden_confounder, with_iv):
        super(CustomDataset, self).__init__()
        self.treatments = data['treatments']
        self.outcomes = data['outcomes']
        self.covariates = data['covariates']
        #self.previous_treatment = data['previous_treatments']
        self.ivs = data['generate_ivs']
        self.hidden_confounders = data['generate_hidden_confounders']
        self.with_hidden_confounder = with_hidden_confounder
        self.with_iv = with_iv
        print('with_hidden_confounder', with_hidden_confounder)
        print('with_iv', with_iv)
    def __getitem__(self, index):
        treatments = self.treatments[index, 1:, :]
        outcome = self.outcomes[index, 1:, :]
        confounders = self.covariates[index, 1:, :]
        previous_treatments = self.treatments[index, :-1, :]
        #previous_treatments = self.previous_treatment[index, :, :]
        if self.with_hidden_confounder:
            hidden_confounders = self.hidden_confounders[index, 1:, :]
            confounders = np.concatenate([confounders, hidden_confounders], axis=-1)
        if self.with_iv:
            ivs = self.ivs[index, 1:, :]
        else:
            ivs = np.zeros((1,))
        return previous_treatments, confounders, treatments, ivs, outcome
    
    def __len__(self):
        return len(self.treatments)    



def init_args():
    parser = argparse.ArgumentParser(description='RMSN for time-series forcasting')
    parser.add_argument('--batch_size', type=int, default=1024, metavar='G',
                        help='batch size for train')
    parser.add_argument('--propensity_hidden_size', type=int, default=16, metavar='G',
                        help='propensity network hidden size for lstm')            
    parser.add_argument('--seq_hidden_size', type=int, default=32, metavar='G',
                        help='sequence hidden size')
    parser.add_argument('--fc_hidden_size', type=int, default=32, metavar='G',
                        help='full connect hidden size')
    parser.add_argument('--num_layer', type=int, default=1, metavar='G',
                        help='lstm layer number')
    parser.add_argument('--total_sequence', type=int, default=16, metavar='G',
                        help='total data sequence length')
    parser.add_argument('--sequence_length', type=int, default=14, metavar='G',
                        help='data sequence length')
    parser.add_argument('--projection_horizon', type=int, default=1, metavar='G',
                        help='projection horizon for prediction')
    parser.add_argument('--encoder_hidden_size', type=int, default=16, metavar='G',
                        help='encoder lstm hidden size')
    parser.add_argument('--encoder_epoch', type=int, default=20, metavar='G',
                        help='the number of encoder train epoch')
                        #310
    parser.add_argument('--decoder_epoch', type=int, default=20, metavar='G',
                        help='the number of decoder train epoch')
                        #3: 310 epoch optimal
                        #5: 400 epoch optimal
    parser.add_argument('--propensity_epoch', type=int, default=20, metavar='G',
                        help='propensity train epoch')
    parser.add_argument('--propensity_lr', type=float, default=1e-3, metavar='G',
                        help='propensity learning rate')
    parser.add_argument('--encoder_lr', type=float, default=1e-3, metavar='G',
                        help='encoder learning rate')
    parser.add_argument('--decoder_lr', type=float, default=1e-3, metavar='G',
                        help='decoder learning rate')
    parser.add_argument('--with_iv', type=bool, default=True, metavar='G',
                        help='use predicted ivs')
    parser.add_argument('--with_confounder', type=bool, default=False, metavar='G',
                        help='use predicted hidden confounder')
    parser.add_argument("--num_hidden_confounders", default=10, type=int)
    parser.add_argument("--num_ivs", default=10, type=int)
    parser.add_argument("--num_treatments", default=1, type=int)
    parser.add_argument('--random_seed', type=int, default=40, metavar='G',
                        help='random seed')
    parser.add_argument('--cuda_id', type=int, default=0, metavar='G',
                        help='GPU ID')
    parser.add_argument('--confounding_degree', type=float, default=0.9, metavar='G',
                        help='confounding degree for synthetic data')  
    parser.add_argument('--iv_degree', type=float, default=0.5, metavar='G',
                        help='iv degree for synthetic data')  
    return parser.parse_args()    




if __name__=='__main__':


    args = init_args()


    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    treatment_size = 1
    outcome_size = 1
    iv_size = args.num_ivs
    
    file_name = 'results_new/all_simulate_data_with_predicted_variables_15_{}_{}_{}_new.pkl'.format(args.random_seed, args.confounding_degree, args.iv_degree)
    print('file_name', file_name)
    data = joblib.load(file_name)
    if args.with_confounder == True:
        confounder_size =  data['covariates'].shape[-1] + args.num_hidden_confounders
    else:
        confounder_size =  data['covariates'].shape[-1]
    print('confounder_size', confounder_size)

    train_dict = dict()
    val_dict = dict()
    test_dict = dict()

    for key in data.keys():
        if key != 'sequence_length':
            train_dict[key] = data[key][:8000,:,:]
            val_dict[key] = data[key][8000:9000,:,:]
            test_dict[key] = data[key][9000:,:,:]


    train_data = CustomDataset(train_dict, with_hidden_confounder=args.with_confounder, with_iv=args.with_iv)
    val_data= CustomDataset(val_dict, with_hidden_confounder=args.with_confounder, with_iv=args.with_iv)
    test_data = CustomDataset(test_dict, with_hidden_confounder=args.with_confounder, with_iv=args.with_iv)



    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    #propensity train process
    if args.with_iv:
        propensity_num = Propensity_Net_Num(input_size=treatment_size+iv_size, hidden_size=args.propensity_hidden_size, num_treatments=treatment_size, num_layers=args.num_layer, args=args).cuda(args.cuda_id)
    else:
        propensity_num = Propensity_Net_Num(input_size=treatment_size, hidden_size=args.propensity_hidden_size, num_treatments=treatment_size, num_layers=args.num_layer, args=args).cuda(args.cuda_id)
    
    train_propensity_net(args, propensity_num, train_loader, numerator=True)

    if args.with_iv:
        propensity_den = Propensity_Net_Den(input_size=treatment_size+confounder_size+iv_size, hidden_size=args.propensity_hidden_size, num_treatments=treatment_size, num_layers=args.num_layer, args=args).cuda(args.cuda_id)
    else:
        propensity_den = Propensity_Net_Den(input_size=treatment_size+confounder_size, hidden_size=args.propensity_hidden_size, num_treatments=treatment_size, num_layers=args.num_layer, args=args).cuda(args.cuda_id)
    

    train_propensity_net(args, propensity_den, train_loader, numerator=False)

    propensity_num.eval()
    propensity_den.eval()


    #outcome prediction
    outcome_encoder = Encoder(input_size=confounder_size+treatment_size, hidden_size=args.encoder_hidden_size, y_size=outcome_size, num_layers=1, args=args).cuda(args.cuda_id)
    train_encoder(train_loader, val_loader, outcome_encoder, args, propensity_num, propensity_den)
    outcome_encoder.eval()


    outcome_decoder = Decoder(input_size=treatment_size, hidden_size=args.encoder_hidden_size, y_size=outcome_size, num_layers=1).cuda(args.cuda_id)
    train_decoder(train_loader, val_loader, outcome_decoder, outcome_encoder, args, propensity_num, propensity_den)
    outcome_decoder.eval()

    test_rmse, test_wmape = test_evaluation(test_loader, outcome_encoder, outcome_decoder, args)
    print('test rmse:', test_rmse)
    print('test wmape', test_wmape)
   





