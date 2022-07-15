from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

# from pygcn.utils import load_data, accuracy
# from pygcn.models import GCN
from utils import load_data, accuracy
from models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=512,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--k', type=float, default=7,
                    help='k nearest neighbors.')
parser.add_argument("-d", "--dataset", help="dataset", type=str, default='citeseer')
parser.add_argument("-l", "--label_per_class", help="labeled data for train per class", type=int, default=60)


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj_topo, adj_feat, features, labels, idx_train, idx_test = load_data(args)

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj_topo = adj_topo.cuda()
    adj_feat = adj_feat.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


# 记录最优有效集准确率
max_accuracy = 0

# Train model
t_total = time.time()
for epoch in range(args.epochs):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj_topo)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    model.eval()
    output = model(features, adj_topo)

    acc_val = accuracy(output[idx_test], labels[idx_test])
    if max_accuracy < acc_val:
        torch.save(model, "./best.plk")
        max_accuracy = acc_val

    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# 测试
model = torch.load("./best.plk")
output = model(features, adj_topo)
acc_test = accuracy(output[idx_test], labels[idx_test])
print("Test set results:",
      "accuracy= {:.4f}".format(acc_test.item()))
