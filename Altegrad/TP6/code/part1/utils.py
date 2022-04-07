"""
Deep Learning on Graphs - ALTEGRAD - Jan 2022
"""

import scipy.sparse as sp
import numpy as np
import torch
import torch.nn as nn

def normalize_adjacency(A):
    #################
    n = A.shape[0]
    A_self_loops = A + sp.identity(n)
    degrees = A_self_loops @ np.ones(n)
    inv_degrees = np.power(degrees, -1)
    inv_D = sp.diags(inv_degrees)
    A_normalized = inv_D @ A_self_loops
    return A_normalized


def sparse_to_torch_sparse(M):
    """Converts a sparse SciPy matrix to a sparse PyTorch tensor"""
    M = M.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((M.row, M.col)).astype(np.int64))
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def loss_function(z, adj, device):
    mse_loss = nn.MSELoss()

    ############## Task 3
    sigmoid = nn.Sigmoid()

    y = list()
    y_pred = list()

    indices = adj._indices()
    n = indices.size(1) 

    y.append(torch.ones(n).to(device))
    y_pred.append(sigmoid(torch.sum(torch.mul(z[indices[0],:], z[indices[1],:]), dim=1)))

    rand_indices = torch.randint(0, z.size(0), indices.size())
    y.append(torch.zeros(n).to(device))
    y_pred.append(sigmoid(torch.sum(torch.mul(z[rand_indices[0],:], z[rand_indices[1],:]), dim=1)))

    y = torch.cat(y, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    loss = mse_loss(y_pred, y)
    return loss
