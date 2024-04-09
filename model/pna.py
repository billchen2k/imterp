import torch
import torch.nn as nn
import math
from einops import einsum

EPS = 1e-5

# each aggregator is a function taking as input X (B x (S or T) x N x Din), adj (N x N), device
# returning the aggregated value of X (B x (S or T) x N x Din) for each dimension

def aggregate_mean(X, adj, device='cuda'):
    adj_ = torch.sign(adj)
    D = torch.sum(adj_, -1, keepdim=True)

    X_sum = torch.einsum("btji,oj->btoi", [X, adj_])
    X_mean = torch.div(X_sum, D)
    return X_mean


def aggregate_normalized_mean(X, adj, device='cuda'):
    # D^{-1} A * X    i.e. the mean of the neighbours
    D = torch.sum(adj, -1, keepdim=True)

    X_sum = torch.einsum("btji,oj->btoi", [X, adj])
    X_mean = torch.div(X_sum, D)
    return X_mean

def aggregate_d(X, adj, device='cuda'):
    # D^{-1} A * X    i.e. the mean of the neighbours
    (B, ST, N, D) = X.shape
    P = torch.ones([B, ST, N, D]).cuda()
    adj_ = torch.sign(adj)
    D = torch.sum(adj_, -1, keepdim=True)
    #    rD = torch.mul(torch.pow(torch.sum(adj, -1, keepdim=True), -0.5), torch.eye(N, device=device))  # D^{-1/2]
    #    adj = torch.matmul(torch.matmul(rD, adj), rD)
    X_sum = torch.einsum("btji,oj->btoi", [P, adj])
    X_mean = torch.div(X_sum, D)
    return X_mean


def aggregate_d_var(X, adj, device='cuda'):
    # relu(D^{-1} A X^2 - (D^{-1} A X)^2)     i.e.  the variance of the features of the neighbours

    (B, ST, N, D) = X.shape
    P = torch.ones([B, ST, N, 1]).cuda()
    D = torch.sum(adj, -1, keepdim=True)

    X_sum = torch.einsum("btji,oj->btoi", [P * P, adj])
    X_sum = torch.div(X_sum, D)
    X_mean = aggregate_mean(X, adj)  # D^{-1} A X
    var = torch.relu(X_sum - X_mean * X_mean)  # relu(mean_squares_X - mean_X^2)
    return var


def aggregate_d_std(X, adj, device='cuda'):
    # sqrt(relu(D^{-1} A X^2 - (D^{-1} A X)^2) + EPS)     i.e.  the standard deviation of the features of the neighbours
    # the EPS is added for the stability of the derivative of the square root
    std = torch.sqrt(aggregate_d_var(X, adj, device) + EPS)  # sqrt(mean_squares_X - mean_X^2)
    return std


def aggregate_max(X, adj, min_value=-math.inf, device='cuda'):  #softmax is better

    (B, ST, N, _) = X.shape
    adj = adj.unsqueeze(-1)
    X = X.unsqueeze(-2).repeat(1, 1, 1, N, 1).permute([0, 1, 3, 2, 4])
    M = torch.where(adj > 0.0, X, torch.tensor(min_value, device=device))
    max = torch.max(M, -2)[0]
    return max


def aggregate_min(X, adj, max_value=math.inf, device='cuda'):  #softmin is better

    (B, ST, N, _) = X.shape
    adj = adj.unsqueeze(-1)
    X = X.unsqueeze(-2).repeat(1, 1, 1, N, 1).permute([0, 1, 3, 2, 4])
    M = torch.where(adj > 0.0, X, torch.tensor(max_value, device=device))
    min = torch.min(M, -2)[0]
    return min


def aggregate_var(X, adj, device='cuda'):
    # relu(D^{-1} A X^2 - (D^{-1} A X)^2)     i.e.  the variance of the features of the neighbours

    D = torch.sum(adj, -1, keepdim=True)

    X_sum = torch.einsum("btji,oj->btoi", [X * X, adj])
    X_sum = torch.div(X_sum, D)
    X_mean = aggregate_mean(X, adj)  # D^{-1} A X
    var = torch.relu(X_sum - X_mean * X_mean)  # relu(mean_squares_X - mean_X^2)
    return var


def aggregate_std(X, adj, device='cuda'):
    # sqrt(relu(D^{-1} A X^2 - (D^{-1} A X)^2) + EPS)     i.e.  the standard deviation of the features of the neighbours
    # the EPS is added for the stability of the derivative of the square root
    std = torch.sqrt(aggregate_var(X, adj, device) + EPS)  # sqrt(mean_squares_X - mean_X^2)
    return std


def aggregate_sum(X, adj, device='cuda'):
    # A * X    i.e. the mean of the neighbours

    X_sum = torch.einsum("btji,oj->btoi", [X, adj])
    return X_sum


def aggregate_softmax(X, adj, device='cuda'):
    # for each node sum_i(x_i*exp(x_i)/sum_j(exp(x_j)) where x_i and x_j vary over the neighbourhood of the node
    X_sum = torch.einsum("btji,oj->btoi", [X, adj])
    softmax = torch.nn.functional.softmax(X_sum, dim=2)
    return softmax


def aggregate_softmin(X, adj, device='cuda'):
    # for each node sum_i(x_i*exp(-x_i)/sum_j(exp(-x_j)) where x_i and x_j vary over the neighbourhood of the node
    return -aggregate_softmax(-X, adj, device=device)


AGGREGATORS = {
    'mean': aggregate_mean,
    'sum': aggregate_sum,
    'max': aggregate_max,
    'min': aggregate_min,
    'std': aggregate_std,
    'var': aggregate_var,
    'normalized_mean': aggregate_normalized_mean,
    'softmax': aggregate_softmax,
    'softmin': aggregate_softmin,
    'distance': aggregate_d,
    'd_std': aggregate_d_std
}

# each scaler is a function that takes as input X (B x ST x N x Din), adj (B x ST x N x N) and
# avg_d (dictionary containing averages over training set) and returns X_scaled (B x ST x N x Din) as output

def scale_identity(X, adj, deg):
    return X

def scale_amplification(X, adj, deg):
    # log(D + 1) / d * X     where d is the average of the ``log(D + 1)`` in the training set

    D = torch.sum(adj, -1)
    scale = (torch.log(D + 1) / deg).unsqueeze(-1)
    X_scaled = torch.mul(scale, X)
    return X_scaled


def scale_attenuation(X, adj, deg):
    # (log(D + 1))^-1 / d * X     where d is the average of the ``log(D + 1))^-1`` in the training set
    D = torch.sum(adj, -1)
    scale = (deg / torch.log(D + 1)).unsqueeze(-1)
    X_scaled = torch.mul(scale, X)
    return X_scaled


SCALERS = {
    'identity': scale_identity,
    'amplification': scale_amplification,
    'attenuation': scale_attenuation,
}
