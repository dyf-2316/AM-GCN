import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy
from models import AM_GCN

# 设置训练参数
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-3,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden1', type=int, default=768,
                    help='Number of hidden units.')
parser.add_argument('--hidden2', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--k', type=float, default=7,
                    help='k nearest neighbors.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# 设置随机数种子
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

# 加载数据
adj_topo, adj_feat, features, labels, idx_train, idx_val, idx_test = load_data(k=args.k)

# 设置模型
model = AM_GCN(nfeat=features.shape[1],
               nhid1=args.hidden,
               nhid2=args.hidden,
               nclass=labels.max().item() + 1,
               dropout=args.dropout)
# 设置优化器
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj_topo = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train():
    t = time.time()
    model.train()
    optimizer.zero_grad()

    output, z_t, z_f, z_ct, z_cf = model(features, adj_topo, adj_feat)
    acc_train = accuracy(output[idx_train], labels[idx_train])
    # 使用负对数似然函数作为损失函数
    loss_t = F.nll_loss(output[idx_train], labels[idx_train])
    loss_train = loss_t
    loss_train.backward()
    optimizer.step()

    model.eval()
    output, z_t, z_f, z_ct, z_cf = model(features, adj_topo, adj_feat)
    acc_val = accuracy(output[idx_val], labels[idx_val])
    loss_t = F.nll_loss(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_t: {:.4f}'.format(loss_t.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output, z_t, z_f, z_ct, z_cf = model(features, adj_topo, adj_feat)
    loss_t = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_t.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# 训练
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# 测试
test()
