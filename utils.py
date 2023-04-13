import torch
import torch.nn as nn
import numpy as np

class AutoRegressiveLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.1):
        super(AutoRegressiveLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.cell = nn.LSTMCell(self.input_size+self.output_size, self.hidden_size)
        self.fc = nn.Sequential(nn.Linear(in_features=self.hidden_size,out_features=self.output_size),
                                nn.Tanh())
        self.dropout_rate = dropout_rate
    #variational dropout autoregression lstm
    def forward(self, inputs, initial_state=None):
        time_steps = inputs.size(0)
        outputs = []
        # if initial_state:
        h, c, z = initial_state
        out_dropout = torch.bernoulli(z.data.new(z.data.size()).fill_(1 - self.dropout_rate)) / (1 - self.dropout_rate)
        h_dropout = torch.bernoulli(h.data.new(h.data.size()).fill_(1 - self.dropout_rate)) / (1 - self.dropout_rate)
        c_dropout = torch.bernoulli(h.data.new(h.data.size()).fill_(1 - self.dropout_rate)) / (1 - self.dropout_rate)
        for t in range(time_steps):
            combine_input = torch.cat([inputs[t,:,:], z],dim=-1)
            h, c = self.cell(combine_input, (h,c))
            z = self.fc(h)
            if self.cell.training:
                z = z * out_dropout
                h, c = h * h_dropout, c * c_dropout
            outputs.append(z)
        return torch.stack(outputs, dim=0), (h,c)



def compute_sequence_length(sequence):
    used = torch.sign(torch.max(torch.abs(sequence), dim=-1)[0])
    length = torch.sum(used, dim=0)
    return length


class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples  
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
    '''
    def __init__(self, x_dim, y_dim, hidden_size=64):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        #print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def forward(self, x_samples, y_samples): 
        mu, logvar = self.get_mu_logvar(x_samples)
        
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 /2./logvar.exp()  
        
        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 

        return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)



class InfoNCE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size=64):
        super(InfoNCE, self).__init__()
        self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1),
                                    nn.Softplus())
    
    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self.F_func(torch.cat([x_samples,y_samples], dim = -1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim = -1))  #[sample_size, sample_size, 1]

        lower_bound = T0.mean() - (T1.logsumexp(dim = 1).mean() - np.log(sample_size)) 
        return lower_bound

    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)