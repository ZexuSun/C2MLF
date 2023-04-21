from configparser import ConfigParser
import torch
import torch.nn as nn
import numpy as np
import os
from torch.distributions import Normal
from utils import AutoRegressiveLSTM, InfoNCE, CLUB



class FactorModel(nn.Module):
    def __init__(self, params, hyperparams, args):
        super(FactorModel, self).__init__()
        self.num_treatments = params['num_treatments']
        self.num_covariates = params['num_covariates']
        self.num_raw_covariates = params['num_raw_covariates']
        self.num_confounders = params['num_confounders']
        self.num_ivs = params['num_ivs']
        self.num_outcomes = params['num_outcomes']
        self.max_sequence_length = params['max_sequence_length']
        self.num_epochs = params['num_epochs']
        self.lstm_hidden_units = hyperparams['lstm_hidden_units']
        self.fc_hidden_units = hyperparams['fc_hidden_units']
        self.learning_rate = hyperparams['learning_rate']
        self.batch_size = hyperparams['batch_size']
        self.args = args        
        #print('self.num_covariates', self.num_covariates)
        #print('self.num_treatments', self.num_treatments)
        self.trainable_init_input_confounder = nn.Parameter(torch.zeros(1, 1, 
                                                self.num_covariates+self.num_treatments)).cuda(args.cuda_id)
        self.trainable_init_input_iv = nn.Parameter(torch.Tensor(1, 1, 
                                                self.num_covariates+self.num_treatments)).cuda(args.cuda_id)
        self.trainable_h0_confounder, self.trainable_c0_confounder, self.trainable_z0_confounder = self.trainable_init_h_confounder()
        self.trainable_h0_iv, self.trainable_c0_iv, self.trainable_z0_iv = self.trainable_init_h_iv()

        #independent mutual information which includes the loss function
        self.term_a = InfoNCE(self.num_covariates+self.num_confounders+self.num_ivs, self.num_treatments)
        self.term_b = InfoNCE(self.num_covariates+self.num_confounders+self.num_treatments, self.num_outcomes)
        self.term_c_1 = CLUB(self.num_confounders, self.num_ivs)
        self.term_c_2 = CLUB(self.num_covariates, self.num_ivs)
        self.term_d = CLUB(self.num_outcomes+self.num_treatments+self.num_covariates+self.num_confounders,\
                           self.num_ivs+self.num_treatments+self.num_covariates+self.num_confounders)


        #confounder lstm generation
        self.lstm_confounder = AutoRegressiveLSTM(input_size=self.num_covariates + self.num_treatments,
                            hidden_size=self.lstm_hidden_units, output_size=self.num_confounders).cuda(args.cuda_id)
        #iv lstm generation
        self.lstm_iv = AutoRegressiveLSTM(input_size=self.num_covariates + self.num_treatments,
                            hidden_size=self.lstm_hidden_units, output_size=self.num_ivs).cuda(args.cuda_id)

    

        #prediction for each treatment
        self.confounder_fc_1 = nn.Sequential(nn.Linear(self.num_covariates + self.num_confounders, self.fc_hidden_units),
                                 nn.ELU(),
                                 nn.Linear(self.fc_hidden_units, self.fc_hidden_units // 2),
                                 nn.ELU(),
                                 nn.Linear(self.fc_hidden_units // 2, 1)).cuda(args.cuda_id)
        self.confounder_fc_2 = nn.Sequential(nn.Linear(self.num_covariates + self.num_confounders, self.fc_hidden_units),
                                 nn.ELU(),
                                 nn.Linear(self.fc_hidden_units, self.fc_hidden_units // 2),
                                 nn.ELU(),
                                 nn.Linear(self.fc_hidden_units // 2, 1)).cuda(args.cuda_id)

        self.iv_fc_1 = nn.Sequential(nn.Linear(self.num_covariates + self.num_ivs, self.fc_hidden_units),
                                 nn.ELU(),
                                 nn.Linear(self.fc_hidden_units, self.fc_hidden_units // 2),
                                 nn.ELU(),
                                 nn.Linear(self.fc_hidden_units // 2, 1)).cuda(args.cuda_id) 
        self.iv_fc_2 = nn.Sequential(nn.Linear(self.num_covariates + self.num_ivs, self.fc_hidden_units),
                                 nn.ELU(),
                                 nn.Linear(self.fc_hidden_units, self.fc_hidden_units // 2),
                                 nn.ELU(),
                                 nn.Linear(self.fc_hidden_units // 2, 1)).cuda(args.cuda_id) 


        self.confounder_decoders = [self.confounder_fc_1, self.confounder_fc_2]
        self.iv_decoders = [self.iv_fc_1, self.iv_fc_2]

    def trainable_init_h_confounder(self):
        h0 = torch.zeros(1, self.lstm_hidden_units)
        c0 = torch.zeros(1, self.lstm_hidden_units)
        z0 = torch.zeros(1, self.num_confounders)
        trainable_h0 = nn.Parameter(h0, requires_grad=True)
        trainable_c0 = nn.Parameter(c0, requires_grad=True)
        trainable_z0 = nn.Parameter(z0, requires_grad=True)
        return trainable_h0, trainable_c0, trainable_z0

    def trainable_init_h_iv(self):
        h0 = torch.zeros(1, self.lstm_hidden_units)
        c0 = torch.zeros(1, self.lstm_hidden_units)
        z0 = torch.zeros(1, self.num_ivs)
        trainable_h0 = nn.Parameter(h0, requires_grad=True)
        trainable_c0 = nn.Parameter(c0, requires_grad=True)
        trainable_z0 = nn.Parameter(z0, requires_grad=True)
        return trainable_h0, trainable_c0, trainable_z0

    def forward(self, previous_covariates, previous_treatments, current_covariates):
        batch_size = previous_covariates.size(0)
        previous_covariates_and_treatments = torch.cat([previous_covariates, previous_treatments], -1).permute(1, 0, 2)
        lstm_input_confounder = torch.cat([self.trainable_init_input_confounder.repeat(1, batch_size, 1), previous_covariates_and_treatments], dim=0)
        lstm_input_confounder = lstm_input_confounder.float()
        lstm_input_iv = torch.cat([self.trainable_init_input_confounder.repeat(1, batch_size, 1), previous_covariates_and_treatments], dim=0)
        lstm_input_iv = lstm_input_iv.float()
        #generate confouder and iv

        lstm_output_confounder, _ = self.lstm_confounder(lstm_input_confounder, initial_state=(self.trainable_h0_confounder.repeat(batch_size, 1).cuda(self.args.cuda_id),
                                              self.trainable_c0_confounder.repeat(batch_size, 1).cuda(self.args.cuda_id),
                                              self.trainable_z0_confounder.repeat(batch_size, 1).cuda(self.args.cuda_id)))
        lstm_output_iv, _ = self.lstm_iv(lstm_input_confounder, initial_state=(self.trainable_h0_iv.repeat(batch_size, 1).cuda(self.args.cuda_id),
                                              self.trainable_c0_iv.repeat(batch_size, 1).cuda(self.args.cuda_id),
                                              self.trainable_z0_iv.repeat(batch_size, 1).cuda(self.args.cuda_id)))

        #definition of confounders1 and iv1
        hidden_confounders = lstm_output_confounder.view(-1, self.num_confounders)
        ivs = lstm_output_iv.view(-1, self.num_ivs)
        current_covariates = current_covariates.reshape(-1, self.num_covariates)



        multitask_input_confounder = torch.cat([hidden_confounders, current_covariates], dim=-1).float()
        multitask_input_iv = torch.cat([ivs, current_covariates], dim=-1).float()
        confounder_pred_treatments = []
        iv_pred_treatments = []
        for treatment in range(self.num_treatments):
            confounder_pred_treatments.append(self.confounder_decoders[treatment](multitask_input_confounder))
            iv_pred_treatments.append(self.iv_decoders[treatment](multitask_input_iv))
        confounder_pred_treatments = torch.cat(confounder_pred_treatments, dim=-1).float()
        iv_pred_treatments = torch.cat(iv_pred_treatments, dim=-1).float()
        return confounder_pred_treatments.view(-1, self.num_treatments), iv_pred_treatments.view(-1, self.num_treatments), \
               hidden_confounders.view(-1, self.num_confounders), ivs.view(-1, self.num_ivs)#hidden_confounders.view(self.max_sequence_length, -1, self.num_confounders), ivs.view(self.max_sequence_length, -1, self.num_ivs)


   

    #inference predict confounder and iv
    def compute_hidden_confounders_and_ivs(self, allloader):
        confounders = []
        ivs = []
        self.eval()
        for i, (previous_covariates, previous_treatments, covariates, treatments, outcomes) in enumerate(allloader):
            previous_covariates, previous_treatments, covariates, treatments, outcomes = \
                previous_covariates.cuda(self.args.cuda_id), previous_treatments.cuda(self.args.cuda_id), \
                covariates.cuda(self.args.cuda_id), treatments.cuda(self.args.cuda_id), outcomes.cuda(self.args.cuda_id)
            _, _, confounder, iv = self.forward(previous_covariates, previous_treatments, covariates)
            confounder = confounder.reshape(-1, self.num_confounders)
            confounder = confounder.cpu().detach().numpy()
            confounders.append(confounder)
            iv = iv.reshape(-1, self.num_ivs)
            iv = iv.cpu().detach().numpy()
            ivs.append(iv)
        #print('len(confounders)', len(confounders))
        #print('len(np.concatenate(confounders, axis=0))', len(np.concatenate(confounders, axis=0)))
        return np.concatenate(confounders, axis=0), np.concatenate(ivs, axis=0)

    
