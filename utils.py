from sklearn.neighbors import NearestNeighbors

import numpy as np
import scipy.sparse as sp
import torch


def load_data(path='../data/cora', dataset='cora'):
    """导入数据集"""
    print('Loading {} dataset...'.format(dataset))
    # 导入节点、特征、标签
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    # 用行压缩矩阵csr_matrix来存储节点特征
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # 构建图拓扑结构
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    # 导入边关系
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    # 用COOrdinate格式来存储拓扑空间的邻接矩阵
    adj_topo = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                             shape=(labels.shape[0], labels.shape[0]),
                             dtype=np.float32)
    # 将拓扑空间邻接矩阵进行对称化
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_topo = adj_topo + (adj_topo.T > adj_topo)

    # 利用knn求解特征空间邻接矩阵
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)
    adj_feat = neigh.kneighbors_graph()

    # 对节点特征以及邻接矩阵进行归一化处理
    features = normalize(features, method='row')
    adj_topo = normalize(adj_topo + sp.eye(adj_topo.shape[0]))
    adj_feat = normalize(adj_feat + sp.eye(adj_feat.shape[0]))

    # TODO：why？
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    # features在tensor中仍以稠密矩阵存储
    features = torch.FloatTensor(np.array(features.todense()))
    # label按类别下标存储
    labels = torch.LongTensor(np.where(labels)[1])
    # adj按稀疏矩阵存储
    adj_topo = sparse_mx_to_torch_sparse_tensor(adj_topo)
    adj_feat = sparse_mx_to_torch_sparse_tensor(adj_feat)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj_topo, adj_feat, features, labels, idx_train, idx_val, idx_test


def encode_onehot(labels):
    """将分类标签转化为one-hot向量"""
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def normalize(mx, method='all'):
    """对矩阵进行归一化操作"""
    if method == 'all':
        row_sum = np.array(mx.sum(1))
        col_sum = np.array(mx.sum(0))
        r_inv = np.power(row_sum, -1 / 2).flatten()
        c_inv = np.power(col_sum, -1 / 2).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        c_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        c_mat_inv = sp.diags(c_inv)
        mx = r_mat_inv.dot(mx.dot(c_mat_inv))
    elif method == 'row':
        row_sum = np.array(mx.sum(1))
        r_inv = np.power(row_sum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """将scipy稀疏矩阵转化为torch的稀疏tensor"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
