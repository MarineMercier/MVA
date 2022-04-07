"""
Graph-based Recommentations - ALTEGRAD - Jan 2022
"""

import pickle
import urllib.request
import numpy as np
from scipy.sparse import lil_matrix, block_diag, diags
from math import ceil

import torch


def load_dataset():
    with open('../data/diginetica_train.p', 'rb') as f:
        data_train = pickle.load(f)
    sessions_train = data_train[0]
    y_train = np.array(data_train[1])

    with open('../data/diginetica_test.p', 'rb') as f:
        data_test = pickle.load(f)
    sessions_test = data_test[0]
    y_test = np.array(data_test[1])

    ############## Task 8
    
    ##################
    # your code here #
    ##################

    print("Number of training sessions", len(sessions_train))
    print("Number of test sessions", len(sessions_test))
    dataset = set()
    for i in sessions_train:
        dataset.update(i)
    max_item_id = max(max(dataset), max(y_train))
    print("Number of uniques items in dataset", len(dataset))
    print("Max Id", max(dataset), "Min Id", min(dataset))
    print("Lenght of longest session", max([len(i) for i in sessions_test]))
    return sessions_train, sessions_test, y_train, y_test, max_item_id


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def generate_batches(sessions, y, batch_size, device, shuffle=False):
    N = len(sessions)
    if shuffle:
        index = np.random.permutation(N)
    else:
        index = np.array(range(N), dtype=np.int32)

    n_batches = ceil(N/batch_size)
    
    adj_lst = list()
    items_lst = list()
    last_item_lst = list()
    idx_lst = list()
    targets_lst = list()
    
    for i in range(0, N, batch_size):
        n_nodes = 0
        for j in range(i, min(i+batch_size, N)):
            n_nodes += np.unique(sessions[index[j]]).size
            
        adj = list()
        items = np.zeros(n_nodes, dtype=np.int32)
        last_item = np.zeros(min(i+batch_size, N)-i, dtype=np.int32)
        idx = np.zeros(n_nodes, dtype=np.int32)
        targets = np.zeros(min(i+batch_size, N)-i, dtype=np.int32)
        
        node_count = 0
        for j in range(i, min(i+batch_size, N)):
            nodes = np.unique(sessions[index[j]])
            node_to_idx = {nodes[i]:i for i in range(nodes.size)}
            A = lil_matrix((nodes.size, nodes.size))
            for k in range(nodes.size):
                A[k,k] = 1
            for k in range(len(sessions[index[j]])-1):
                u = node_to_idx[sessions[index[j]][k]]
                v = node_to_idx[sessions[index[j]][k+1]]
                A[v,u] += 1
                A[u,v] += 1
            
            adj.append(normalize(A))
            items[node_count:node_count+nodes.size] = nodes
            last_item[j-i] = node_count+node_to_idx[sessions[index[j]][-1]]
            idx[node_count:node_count+nodes.size] = j-i
            targets[j-i] = y[index[j]]
            
            node_count += nodes.size
        
        adj = block_diag(adj)
        
        adj_lst.append(sparse_mx_to_torch_sparse_tensor(adj).to(device))
        items_lst.append(torch.LongTensor(items).to(device))
        last_item_lst.append(torch.LongTensor(last_item).to(device))
        idx_lst.append(torch.LongTensor(idx).to(device))
        targets_lst.append(torch.LongTensor(targets).to(device))
   
    return adj_lst, items_lst, last_item_lst, idx_lst, targets_lst