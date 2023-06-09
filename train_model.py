from factor_model import FactorModel
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import torch.optim as optim
#device = 'cuda'


def train_model(factor_model, trainloader, valloader, args):
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(factor_model.parameters(), lr=factor_model.learning_rate)
    for epoch in tqdm(range(factor_model.num_epochs)):
        factor_model.train()
        train_losses = []
        for i, (previous_covariates, previous_treatments, covariates, treatments, outcomes) in enumerate(trainloader):
            previous_covariates, previous_treatments, covariates, treatments, outcomes = \
                previous_covariates.cuda(args.cuda_id), previous_treatments.cuda(args.cuda_id), \
                covariates.cuda(args.cuda_id), treatments.cuda(args.cuda_id), outcomes.cuda(args.cuda_id)
            
            confounder_pred_treatments, iv_pred_treatments, confounders, ivs = factor_model(previous_covariates, previous_treatments, covariates)
            
            #for reshape
            treatment_targets = treatments.reshape(-1, factor_model.num_treatments).float()
            covariates = covariates.reshape(-1, factor_model.num_covariates).float()
            outcomes = outcomes.reshape(-1, 1).float()
            confounders = confounders.reshape(-1, factor_model.num_confounders).float()
            ivs = ivs.reshape(-1, factor_model.num_ivs).float()

            #term_a
            loss_a = factor_model.term_a(torch.cat([covariates, confounders, ivs], dim=-1), treatment_targets)
            #term_b
            loss_b = factor_model.term_b(torch.cat([covariates, confounders, treatment_targets], dim=-1), outcomes)
            #term_c
            loss_c = factor_model.term_c_1(confounders, ivs) + factor_model.term_c_2(covariates, ivs)
            #term_d
            loss_d = factor_model.term_d(torch.cat([outcomes, treatment_targets, covariates, confounders], dim=-1),
                                         torch.cat([ivs, treatment_targets, covariates, confounders], dim=-1))

            #total loss
            train_loss = -loss_a - loss_b + args.alpha * loss_c + args.beta * loss_d
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.item())

            
            #factor_train_loss = sum(train_losses) / len(train_losses)
        factor_train_loss = sum(train_losses) / len(train_losses)

        factor_model.eval()
        val_losses = []
        with torch.no_grad():
            for i, (previous_covariates, previous_treatments, covariates, treatments, outcomes) in enumerate(valloader):
                previous_covariates, previous_treatments, covariates, treatments, outcomes = \
                    previous_covariates.cuda(args.cuda_id), previous_treatments.cuda(args.cuda_id), \
                    covariates.cuda(args.cuda_id), treatments.cuda(args.cuda_id), outcomes.cuda(args.cuda_id)

                confounder_pred_treatments, iv_pred_treatments, confounders, ivs = factor_model(previous_covariates, previous_treatments, covariates)

                #for reshape
                treatment_targets = treatments.reshape(-1, factor_model.num_treatments).float()
                covariates = covariates.reshape(-1, factor_model.num_covariates).float()
                outcomes = outcomes.reshape(-1, 1).float()
                confounders = confounders.reshape(-1, factor_model.num_confounders).float()
                ivs = ivs.reshape(-1, factor_model.num_ivs).float()

                   
                #term_a
                loss_a = factor_model.term_a(torch.cat([covariates, confounders, ivs], dim=-1), treatment_targets)
                #term_b
                loss_b = factor_model.term_b(torch.cat([covariates, confounders, treatment_targets], dim=-1), outcomes)
                #term_c
                loss_c = factor_model.term_c_1(confounders, ivs) + factor_model.term_c_2(covariates, ivs)
                #term_d
                loss_d = factor_model.term_d(torch.cat([outcomes, treatment_targets, covariates, confounders], dim=-1),
                                         torch.cat([ivs, treatment_targets, covariates, confounders], dim=-1))

                #total loss
                val_loss = -loss_a - loss_b + args.alpha * loss_c + args.beta * loss_d

                val_losses.append(val_loss)
            factor_val_loss = sum(val_losses) / len(val_losses)
        print("epoch {} ---- train_loss:{:.5f} val_loss:{:.5f}".format(epoch+1, factor_train_loss, factor_val_loss))
