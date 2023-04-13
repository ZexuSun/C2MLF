from tkinter.tix import InputOnly
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions import Normal
import math
from tqdm import tqdm
# Description: continuous outcome, continuous treatment, continuous or binary covariates
# Data is of shape (n,T,K), where d=1 for outcome, K for treatment, p for time covariates 
# Procedure: estimate IPTW and re-weight data, estimate encoder, and estimate decoder with pre-trained encoder


def calculate_stablized_weight(current_treatment, mu, log_std):
    sigma = torch.exp(log_std)
    pi = torch.tensor(math.pi)
    weight = (1 / (torch.sqrt(2 * pi * sigma) + 1e-8)) * torch.exp((current_treatment - mu) / (2 * sigma**2))
    return weight

class Propensity_Net_Num(nn.Module):
    def __init__(self, input_size, hidden_size, num_treatments, num_layers, args):
        super(Propensity_Net_Num, self).__init__()
        self.args = args
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # -> x needs to be: (batch_size, seq_length, input_size)
        self.fc1 = nn.Linear(hidden_size, num_treatments)
        self.fc2 = nn.Linear(hidden_size, num_treatments)
        self.tanh = nn.Tanh()

    def forward(self, x):         
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda(self.args.cuda_id)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda(self.args.cuda_id)        
        # x: (batch_size, seq_len, input_size), h0: (num_layers, batch_size, hidden_size1)

        x = x.float()
        h0 = h0.float()
        c0 = c0.float()
        
        # Forward propagate RNN
        out, _ = self.lstm(x, (h0,c0))  
        # out: tensor of shape (batch_size, seq_length, hidden_size1) - Outputs are hidden states
        
        x = self.tanh(out)
        mu = self.fc1(x)
        x = self.fc2(x)
        log_std = self.tanh(x)
        # out: tensor of shape (batch_size, seq_length, num_treatments), the Gaussion distribution of output
        return mu, log_std

class Propensity_Net_Den(nn.Module):
    def __init__(self, input_size, hidden_size, num_treatments, num_layers, args):
        super(Propensity_Net_Den, self).__init__()
        self.args = args
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(400, 5)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # -> x needs to be: (batch_size, seq_length, input_size)
        self.fc1 = nn.Linear(hidden_size, num_treatments)
        self.fc2 = nn.Linear(hidden_size, num_treatments)
        self.tanh = nn.Tanh()

    def forward(self, x, c):    
        #city_emb = self.embed(torch.tensor(c[:,:, 0]).long())
        #c = torch.cat([city_emb, c[:,:,1:]], dim=-1)     
        x = torch.cat((x, c), dim=-1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda(self.args.cuda_id)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda(self.args.cuda_id)        
        # x: (batch_size, seq_len, input_size), h0: (num_layers, batch_size, hidden_size1)

        # Forward propagate RNN
        x = x.float()
        h0 = h0.float()
        c0 = c0.float()
        
        out, _ = self.lstm(x, (h0,c0))  
        # out: tensor of shape (batch_size, seq_length, hidden_size1) - Outputs are hidden states
        
        x = self.tanh(out)
        mu = self.fc1(x)
        x = self.fc2(x)
        log_std = self.tanh(x)
        # out: tensor of shape (batch_size, seq_length, num_treatments), the Gaussion distribution of output
        return mu, log_std


def train_propensity_net(args, peopensity_network, train_loader, numerator=True):
    #train data prepare
    optimizer = optim.Adam(peopensity_network.parameters(), lr=args.propensity_lr) 
    for epoch in tqdm(range(args.propensity_epoch)):
        losses_propensity = []

        for i, (previous_treatments, current_confounders, current_treatments, current_ivs, outcomes) in enumerate(train_loader): 
            #treatment predict label
            previous_treatments, current_confounders, current_treatments, current_ivs, outcomes = previous_treatments.float().cuda(args.cuda_id), current_confounders.float().cuda(args.cuda_id),\
                 current_treatments.float().cuda(args.cuda_id), current_ivs.float().cuda(args.cuda_id), outcomes.float().cuda(args.cuda_id)
            if args.with_iv:
                propensity_input = torch.cat([previous_treatments, current_ivs], dim=-1)            
            else:
            #propensity num input A_train_t-1 and Z_train_t
                propensity_input = previous_treatments

            if numerator:
                mu, log_std = peopensity_network(propensity_input)
            else:
                mu, log_std = peopensity_network(propensity_input, current_confounders)
            # print(mu.shape, log_std.shape, 'mushape, stdshape')
            treatment_dist = Normal(mu, torch.exp(log_std))
            # print(treatment_dist, current_treatments.shape)
            loss_propensity = -torch.mean(treatment_dist.log_prob(current_treatments))
            losses_propensity.append(loss_propensity.item())
            optimizer.zero_grad()
            loss_propensity.backward()
            optimizer.step()
            
            
        # if (epoch+1) % (args.propensity_epoch) == 0 and numerator == True:
        if numerator == True:
            print (f'Propensity_num Epoch [{epoch+1}/{args.propensity_epoch}], Loss: {sum(losses_propensity)/len(losses_propensity):.4f}')
        if numerator == False:
            print (f'Propensity_den Epoch [{epoch+1}/{args.propensity_epoch}], Loss: {sum(losses_propensity)/len(losses_propensity):.4f}') 



def get_stabilized_weight(args, propensity_net_num, propensity_net_den, previous_treatments, current_ivs, current_treatments, current_confounders, sequence_length, prediction_steps):
    # sample_size = previous_treatments.shape[0]
    tau_max = prediction_steps #for encoder tau_max=1, for decoder tau_max=Tau
    stabilized_weight = torch.ones(size=(current_treatments.shape[0], current_treatments.shape[1], 1)).cuda(args.cuda_id)
    if args.with_iv:
        for t in range(sequence_length):
            for tau in range(tau_max):
                if (t+tau) >= sequence_length:
                    break
                # numerator_mu, numerator_logstd = propensity_net_num(torch.cat((previous_treatments[:, t+tau], current_ivs[:, t+tau]), dim=-1).unsqueeze(1))
                # numerator_mu, numerator_logstd = numerator_mu.squeeze(1), numerator_logstd.squeeze(1)
                # weight_numerator = calculate_stablized_weight(current_treatments[:, t+tau], numerator_mu, numerator_logstd)
                
                denominator_mu, denominator_logstd = propensity_net_den(torch.cat((previous_treatments[:, t+tau], current_ivs[:, t+tau]), dim=-1).unsqueeze(1), current_confounders[:, t+tau].unsqueeze(1))
                denominator_mu, denominator_logstd = denominator_mu.squeeze(1), denominator_logstd.squeeze(1)
                weight_denominator = calculate_stablized_weight(current_treatments[:, t+tau], denominator_mu, denominator_logstd)
                # numerator = weight_numerator[:, 0]*weight_numerator[:, 1]
                # numerator = numerator.unsqueeze(1)

                #denominator = weight_denominator[:, 0]*weight_denominator[:, 1]
                #denominator = denominator.unsqueeze(1)
                stabilized_weight[:, t] *= 1 / weight_denominator
    else:
        for t in range(sequence_length):
            for tau in range(tau_max):
                if (t+tau) >= sequence_length:
                    break
                # numerator_mu, numerator_logstd = propensity_net_num(previous_treatments[:, t+tau].unsqueeze(1))
                # numerator_mu, numerator_logstd = numerator_mu.squeeze(1), numerator_logstd.squeeze(1)
                # weight_numerator = calculate_stablized_weight(current_treatments[:, t+tau], numerator_mu, numerator_logstd)
                
                denominator_mu, denominator_logstd = propensity_net_den(previous_treatments[:, t+tau].unsqueeze(1), current_confounders[:, t+tau].unsqueeze(1))
                denominator_mu, denominator_logstd = denominator_mu.squeeze(1), denominator_logstd.squeeze(1)
                weight_denominator = calculate_stablized_weight(current_treatments[:, t+tau], denominator_mu, denominator_logstd)
                # numerator = weight_numerator[:, 0]*weight_numerator[:, 1]
                # numerator = numerator.unsqueeze(1)
                #denominator = weight_denominator[:, 0]*weight_denominator[:, 1]
                #denominator = denominator.unsqueeze(1)
                stabilized_weight[:, t] *= 1 / weight_denominator
            # stabilized_weight[:, t] = stabilized_weight[:, t, 0] * stabilized_weight[:, t, 1]
    numpy_weight = stabilized_weight.detach().cpu().numpy()
    UB = torch.tensor(np.percentile(numpy_weight, 99)).cuda(args.cuda_id)
    LB = torch.tensor(np.percentile(numpy_weight, 1)).cuda(args.cuda_id)
    stabilized_weight = torch.clip(stabilized_weight, min=LB, max=UB)
    return stabilized_weight.cuda(args.cuda_id)
    

                



#Encoder net
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, y_size, num_layers, args):
        super(Encoder, self).__init__()
        self.args = args
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(400, 5)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # -> x needs to be: (batch_size, seq_length, input_size)
        self.fc = nn.Linear(hidden_size, y_size)

    def forward(self, c, x):
        #city_emb = self.embed(torch.tensor(c[:,:, 0]).long())
        #c = torch.cat([city_emb, c[:,:,1:]], dim=-1)     
        x = torch.cat((x, c), dim=-1)
        #print('x.shape', x.shape)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda(self.args.cuda_id)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda(self.args.cuda_id) 
        #print('h0.shape', h0.shape)   
        #print('c0.shape', c0.shape)    
        # x: (batch_size, seq_len, input_size), h0: (num_layers, batch_size, hidden_size1)

        # Forward propagate RNN
        out, _ = self.lstm(x, (h0,c0))  
        # out: tensor of shape (batch_size, seq_length, hidden_size) - Outputs are hidden states
        
        #output hidden state
        h_out = out[:, -1, :] 
        y_out = self.fc(out)
        # out: tensor of shape (batch_size, y_size)

        return y_out, h_out  
def loss_function(y_pred, y_true):
    loss = (y_pred - y_true)**2
    return loss
def train_encoder(train_loader, val_loader, encoder, args, propensity_num, propensity_den):
    optimizer = optim.Adam(encoder.parameters(), lr=args.encoder_lr) 
    for epoch in tqdm(range(args.encoder_epoch)):
        losses_encoder = []
        for i, (previous_treatments, current_confounders, current_treatments, current_ivs, outcomes) in enumerate(train_loader): 
            #treatment predict label
            previous_treatments, current_confounders, current_treatments, current_ivs, outcomes = previous_treatments.float().cuda(args.cuda_id), current_confounders.float().cuda(args.cuda_id),\
                 current_treatments.float().cuda(args.cuda_id), current_ivs.cuda(args.cuda_id), outcomes.cuda(args.cuda_id)

            #encoder_input = torch.cat((current_confounders, current_treatments), dim=-1)
            outcome_prediction, _ = encoder(current_confounders, current_treatments)
            encoder_stabilize_weight = get_stabilized_weight(args, propensity_num, propensity_den, previous_treatments, current_ivs, current_treatments, current_confounders, args.sequence_length + args.projection_horizon, 1)
            normalized_encoder_stabilize_weight = encoder_stabilize_weight / (encoder_stabilize_weight.mean().unsqueeze(0))
            normalized_encoder_stabilize_weight = normalized_encoder_stabilize_weight.detach()
            loss_encoder = torch.mean(normalized_encoder_stabilize_weight * loss_function(outcomes, outcome_prediction))
            optimizer.zero_grad()
            loss_encoder.backward()
            optimizer.step()
            losses_encoder.append(loss_encoder.item())

        with torch.no_grad():
            val_loss_encoder = []
            for i, (previous_treatments, current_confounders, current_treatments, current_ivs, outcomes) in enumerate(val_loader): 
                #treatment predict label
                previous_treatments, current_confounders, current_treatments, outcomes = previous_treatments.float().cuda(args.cuda_id), current_confounders.float().cuda(args.cuda_id),\
                    current_treatments.float().cuda(args.cuda_id), outcomes.float().cuda(args.cuda_id)
                #encoder_input = torch.cat((current_confounders, current_treatments), dim=-1)
                outcome_prediction, _ = encoder(current_confounders, current_treatments)
                loss_encoder = torch.mean(loss_function(outcomes, outcome_prediction))
                val_loss_encoder.append(loss_encoder.item())
            print (f'Encoder Epoch [{epoch+1}/{args.encoder_epoch}], Train Loss: {sum(losses_encoder)/len(losses_encoder):.4f}, Val Loss: {sum(val_loss_encoder)/len(val_loss_encoder):.4f}')

        

#Decoder net

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, y_size, num_layers):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # -> x needs to be: (batch_size, seq_length, input_size)
        self.fc = nn.Linear(hidden_size, y_size)

    def forward(self, x, init_states):
        h0, c0 = init_states, init_states
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)         
        # x: (batch_size, seq_len, input_size), h0: (num_layers, batch_size, hidden_size1)

        # Forward propagate RNN
        out, _ = self.lstm(x, (h0,c0))  
        # out: tensor of shape (batch_size, seq_length, hidden_size) - Outputs are hidden state
        y_out = self.fc(out)
        # out: tensor of shape (batch_size, seq_length, y_size)

        return y_out  



def train_decoder(train_loader, val_loader, outcome_decoder, outcome_encoder, args, propensity_num, propensity_den):
        optimizer = optim.Adam(outcome_decoder.parameters(), lr=args.decoder_lr)
        for epoch in tqdm(range(args.decoder_epoch)):
            losses_decoder = []
            loss_function = nn.MSELoss(reduction='none')
            for i, (previous_treatments, current_confounders, current_treatments, current_ivs, outcomes) in enumerate(train_loader): 
                previous_treatments, current_confounders, current_treatments, current_ivs, outcomes = previous_treatments.float().cuda(args.cuda_id), \
                    current_confounders.float().cuda(args.cuda_id), current_treatments.float().cuda(args.cuda_id), current_ivs.float().cuda(args.cuda_id), outcomes.float().cuda(args.cuda_id)
                #encoder_input = torch.cat((current_confounders[:, :args.sequence_length, :], current_treatments[:, :args.sequence_length, :]), dim=-1)
                decoder_input = current_treatments[:, args.sequence_length:args.sequence_length+args.projection_horizon, :]
                _, init_states = outcome_encoder(current_confounders[:, :args.sequence_length, :], current_treatments[:, :args.sequence_length, :])
                init_states = init_states.detach().unsqueeze(0)
                outcome_prediction = outcome_decoder(decoder_input, init_states)
                stabilized_weight = get_stabilized_weight(args, propensity_num, propensity_den, previous_treatments, current_ivs, current_treatments, current_confounders, args.sequence_length + args.projection_horizon, args.projection_horizon)
                decoder_stabilized_weight = stabilized_weight[:, args.sequence_length:args.sequence_length+args.projection_horizon, :]
                normalized_decoder_stabilized_weight = decoder_stabilized_weight / (decoder_stabilized_weight.mean().unsqueeze(0))
                normalized_decoder_stabilized_weight = normalized_decoder_stabilized_weight.detach()
                loss_decoder = torch.mean(normalized_decoder_stabilized_weight * loss_function(outcome_prediction, outcomes[:, args.sequence_length:args.sequence_length+args.projection_horizon, :]))
                optimizer.zero_grad()   
                loss_decoder.backward()
                optimizer.step()
                losses_decoder.append(loss_decoder.item())
            print(f'Decoder Epoch [{epoch+1}/{args.decoder_epoch}], Loss: {sum(losses_decoder)/len(losses_decoder):.6f}')
        
        with torch.no_grad():
            val_losses_decoder = []
            for i, (previous_treatments, current_confounders, current_treatments, current_ivs, outcomes) in enumerate(val_loader): 
                previous_treatments, current_confounders, current_treatments, outcomes = previous_treatments.float().cuda(args.cuda_id), current_confounders.float().cuda(args.cuda_id),\
                current_treatments.float().cuda(args.cuda_id), outcomes.float().cuda(args.cuda_id)
                #encoder_input = torch.cat((current_confounders[:, :args.sequence_length, :], current_treatments[:, :args.sequence_length, :]), dim=-1)
                decoder_input = current_treatments[:, args.sequence_length:args.sequence_length+args.projection_horizon, :]
                _, init_states = outcome_encoder(current_confounders[:, :args.sequence_length, :], current_treatments[:, :args.sequence_length, :])
                init_states = init_states.detach().unsqueeze(0)
                outcome_prediction = outcome_decoder(decoder_input, init_states)
                loss_decoder = torch.mean(loss_function(outcome_prediction, outcomes[:, args.sequence_length:args.sequence_length+args.projection_horizon, :]))
                val_losses_decoder.append(loss_decoder.item())
            print(f'Decoder Epoch [{epoch+1}/{args.decoder_epoch}], Train Loss: {sum(losses_decoder)/len(losses_decoder):.4f}, Val Loss: {sum(val_losses_decoder)/len(val_losses_decoder):.4f}')

def rmse(y_true, y_pred):
    calculate_rmse = torch.sqrt(torch.mean((y_true - y_pred)**2))
    return calculate_rmse.cpu().detach().numpy()

def wmape(y_pred, y_true):
    wmape_val = torch.sum(torch.abs(y_pred - y_true)) / torch.sum(torch.abs(y_true))
    return wmape_val

def test_evaluation(test_loader, encoder, decoder, args):
    test_rmses = []
    test_wmapes = []
    for i, (previous_treatments, current_confounders, current_treatments, current_ivs, outcomes) in enumerate(test_loader): 
        previous_treatments, current_confounders, current_treatments, outcomes = previous_treatments.float().cuda(args.cuda_id), current_confounders.float().cuda(args.cuda_id),\
            current_treatments.float().cuda(args.cuda_id), outcomes.float().cuda(args.cuda_id)
        # = torch.cat((previous_treatments, current_confounders), dim=-1)
        _, init_states = encoder(current_confounders, previous_treatments)
        init_states = init_states.detach().unsqueeze(0)
        decoder_input = current_treatments[:, args.sequence_length:args.sequence_length+args.projection_horizon, :]
        predicted_outcomes = decoder(decoder_input, init_states)
        batch_rmse = rmse(outcomes[:, args.sequence_length:args.sequence_length+args.projection_horizon, :], predicted_outcomes.detach())
        test_rmses.append(batch_rmse)
        #batch_wmape = wmape(predicted_outcomes.detach() * 0.444 + 0.0839, outcomes[:, args.sequence_length:args.sequence_length+args.projection_horizon, :] * 0.444 + 0.0839)
        batch_wmape = wmape(predicted_outcomes.detach(), outcomes[:, args.sequence_length:args.sequence_length+args.projection_horizon, :])
        test_wmapes.append(batch_wmape)
    test_wmape = sum(test_wmapes) / len(test_wmapes)        
    test_rmse = sum(test_rmses) / len(test_rmses)
    return test_rmse, test_wmape
