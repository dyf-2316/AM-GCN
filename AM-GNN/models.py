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
        z = F.relu(self.gc2(z, adj))
        z = F.dropout(z, self.dropout, training=self.training)
        return z


class Attention(nn.Module):
    def __init__(self, nhid):
        super().__init__()
        self.W_t = nn.Linear(nhid, 16)
        self.W_f = nn.Linear(nhid, 16)
        self.W_c = nn.Linear(nhid, 16)
        self.tanh = nn.Tanh()
        self.q = nn.Linear(16, 1, bias=False)

    def forward(self, z_t, z_f, z_c):
        e_t = self.q(self.tanh(self.W_t(z_t)))
        e_f = self.q(self.tanh(self.W_f(z_f)))
        e_c = self.q(self.tanh(self.W_c(z_c)))
        e = torch.stack((e_t, e_f, e_c), 1).squeeze(-1)
        a = F.softmax(e, dim=1)
        a_t, a_f, a_c = a.chunk(3, 1)
        a_T, a_F, a_C = torch.diag(a_t.squeeze()), torch.diag(a_f.squeeze()), torch.diag(a_c.squeeze())
        z = torch.mm(a_T, z_t) + torch.mm(a_F, z_f) + torch.mm(a_C, z_c)
        return z, a


class AM_GCN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nclass, dropout):
        super().__init__()

        self.gcn_t = GCN(nfeat, nhid1, nhid2, dropout)
        self.gcn_f = GCN(nfeat, nhid1, nhid2, dropout)
        self.gcn_c = GCN(nfeat, nhid1, nhid2, dropout)
        self.attention = Attention(nhid2)
        self.classifier = nn.Sequential(
            nn.Linear(nhid2, nclass),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, adj_topo, adj_feat):
        z_t = self.gcn_t(x, adj_topo)
        z_f = self.gcn_f(x, adj_feat)
        z_ct = self.gcn_c(x, adj_topo)
        z_cf = self.gcn_c(x, adj_feat)
        z_c = (z_cf + z_ct) / 2

        z, attn_weight = self.attention(z_t, z_f, z_c)

        y = self.classifier(z)
        return y, attn_weight, z_t, z_f, z_ct, z_cf, z


