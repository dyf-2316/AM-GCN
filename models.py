import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, dropout):
        super().__init__()

        self.gc1 = GraphConvolution(nfeat, nhid1)
        self.gc2 = GraphConvolution(nhid1, nhid2)
        self.dropout = dropout

    def forward(self, x, adj):
        z = F.relu(self.gc1(x, adj))
        z = F.dropout(z, self.dropout, training=self.training)
        z = self.gc2(z, adj)
        return z


class Attention(nn.Module):
    def __init__(self, nhid):
        super().__init__()
        self.W_t = nn.Linear(nhid, nhid)
        self.W_f = nn.Linear(nhid, nhid)
        self.W_c = nn.Linear(nhid, nhid)
        self.q = nn.Linear(nhid, 1, bias=False)

    def forward(self, z_t, z_f, z_c):
        e_t = self.q(F.tanh(self.W_t(z_t.T)))
        e_f = self.q(F.tanh(self.W_f(z_f.T)))
        e_c = self.q(F.tanh(self.W_c(z_c.T)))
        e = torch.stack((e_t, e_f, e_c, x2), 1)
        a = F.softmax(e, dim=1)
        a_t, a_f, a_c = a.chunk(3, 1)
        return torch.diag(a_t), torch.diag(a_f), torch.diag(a_c)


class AM_GCN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nclass, dropout):
        super().__init__()

        self.gcn_t = GCN(nfeat, nhid1, nhid2, dropout)
        self.gcn_f = GCN(nfeat, nhid1, nhid2, dropout)
        self.gcn_c = GCN(nfeat, nhid1, nhid2, dropout)
        self.attention = Attention(nhid2)
        self.classifier = nn.Linear(nhid2, nclass)

    def forward(self, x, adj_topo, adj_feat):
        z_t = self.gcn_t(x, adj_topo)
        z_f = self.gcn_f(x, adj_feat)
        z_ct = self.gcn_c(x, adj_topo)
        z_cf = self.gcn_c(x, adj_feat)
        z_c = (z_cf + z_ct) / 2
        a_T, a_F, a_C = self.attention(z_t, z_f, z_c)
        z = a_T * z_t + a_F * z_f + a_C * z_c
        y = self.classifier(z)
        return F.softmax(y, dim=1), z_t, z_f, z_ct, z_cf


