import time
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from utils import load_data, metrics, consistency_loss, disparity_loss
from models import AM_GCN


# 设置训练参数
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=123, help='Random seed.')
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-3,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden1', type=int, default=768,
                    help='Number of hidden units.')
parser.add_argument('--hidden2', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--k', type=float, default=7,
                    help='k nearest neighbors.')
parser.add_argument("-d", "--dataset", help="dataset", type=str, default='citeseer')
parser.add_argument("-l", "--label_per_class", help="labeled data for train per class", type=int, default=60)
parser.add_argument('--gamma', type=float, default=0,
                    help='The coefficient of consistency constraint.')
parser.add_argument('--beta', type=float, default=0,
                    help='The coefficient of and disparity constraints.')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# 设置随机数种子
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

# 加载数据
adj_topo, adj_feat, features, labels, idx_train, idx_test = load_data(args)

# 设置模型
model = AM_GCN(nfeat=features.shape[1],
               nhid1=args.hidden1,
               nhid2=args.hidden2,
               nclass=labels.max().item() + 1,
               dropout=args.dropout)
# 设置优化器
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj_topo = adj_topo.cuda()
    adj_feat = adj_feat.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_test = idx_test.cuda()

# 记录最优有效集准确率
max_accuracy = 0

attn_log = []
prefix = "{}_{}_{}_{}_{}".format(args.dataset, args.label_per_class, args.k, args.gamma, args.beta)

# 训练
t_total = time.time()
for epoch in range(args.epochs):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    output, att, z_t, z_f, z_ct, z_cf, z = model(features, adj_topo, adj_feat)
    attn_log.append(att.cpu().detach().numpy())
    acc_train, f1_train = metrics(output[idx_train], labels[idx_train], labels.max().item() + 1)
    # 计算损失函数各部分
    loss_t = F.nll_loss(output[idx_train], labels[idx_train])
    loss_c = consistency_loss(z_ct, z_cf)
    loss_d = disparity_loss(z_t, z_f, z_ct, z_cf)
    # 合成损失函数
    print("loss_t = ", loss_t)
    print("loss_c = ", loss_c)
    print("loss_d = ", loss_d)
    loss_train = loss_t + args.gamma * loss_c + args.beta * loss_d
    loss_train.backward()
    optimizer.step()

    model.eval()
    output, att, z_t, z_f, z_ct, z_cf, z = model(features, adj_topo, adj_feat)

    acc_val, f1_val = metrics(output[idx_test], labels[idx_test], labels.max().item() + 1)
    if max_accuracy < acc_val:
        torch.save(model, "./{}_best.plk".format(prefix))
        max_accuracy = acc_val

    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          "f1_train: {:.4f}".format(f1_train.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          "f1_val: {:.4f}".format(f1_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

np.save("./{}_attn.npy".format(prefix), np.array(attn_log))

# 测试
model = torch.load("./{}_best.plk".format(prefix))
output, att, z_t, z_f, z_ct, z_cf, z = model(features, adj_topo, adj_feat)
acc_test, f1_test = metrics(output[idx_test], labels[idx_test], labels.max().item() + 1)
print("Test set results:",
      "accuracy= {:.4f}".format(acc_test.item()),
      "f1= {:.4f}".format(f1_test.item()))
