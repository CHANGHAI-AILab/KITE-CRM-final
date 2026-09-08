import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import datasetcls
from model import LE
from utils import progress_bar


trainloader = torch.utils.data.DataLoader(datasetcls.train_dataset, batch_size=32, shuffle=True, num_workers=30)
# testloader = torch.utils.data.DataLoader(datasetcls.test_dataset, batch_size=32, shuffle=True, num_workers=30)

testloader_411 = torch.utils.data.DataLoader(datasetcls.test_dataset_411, batch_size=32, shuffle=True, num_workers=30)
# testloader_fuyic = torch.utils.data.DataLoader(datasetcls.test_dataset_fuyic, batch_size=32, shuffle=True, num_workers=30)
# testloader_fzc = torch.utils.data.DataLoader(datasetcls.test_dataset_fzc, batch_size=32, shuffle=True, num_workers=30)
# testloader_renji = torch.utils.data.DataLoader(datasetcls.test_dataset_renji, batch_size=32, shuffle=True, num_workers=30)

net = LE().cuda()
# net = torch.nn.DataParallel(net, device_ids=[0, 1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)


def get_lr(epoch):
    if epoch < 20:
        lr = 0.001
    elif epoch < 40:
        lr = 0.0005
    elif epoch < 60:
        lr = 0.00025
    elif epoch < 80:
        lr = 0.000125
    else:
        lr = 0.00006125
    return lr


def train():
    print("Training Epoch:", str(epoch))
    net.train()
    lr = get_lr(epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (feats0, feats1, feats2, targets) in enumerate(trainloader):
        feats0, feats1, feats2, targets = feats0.cuda(), feats1.cuda(), feats2.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(feats0, feats1, feats2)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        train_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100. * correct/total, correct, total))


def test():
    net.eval()

    test_loss = 0
    correct = 0
    total = 0

    for batch_idx, (feats0, feats1, feats2, targets, names) in enumerate(testloader_411):
        feats0, feats1, feats2, targets = feats0.cuda(), feats1.cuda(), feats2.cuda(), targets.cuda()
        outputs = net(feats0, feats1, feats2)
        loss = criterion(outputs, targets)

        test_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        progress_bar(batch_idx, len(testloader_411), 'Name: %s | Loss: %.3f | Acc: %.3f%% (%d/%d)' % ("411", test_loss/(batch_idx+1), 100. * correct/total, correct, total))

    # for batch_idx, (feats0, feats1, feats2, targets, names) in enumerate(testloader_fuyic):
    #     feats0, feats1, feats2, targets = feats0.cuda(), feats1.cuda(), feats2.cuda(), targets.cuda()
    #     outputs = net(feats0, feats1, feats2)
    #     loss = criterion(outputs, targets)
    #
    #     test_loss += loss.data
    #     _, predicted = torch.max(outputs.data, 1)
    #     total += targets.size(0)
    #     correct += predicted.eq(targets.data).cpu().sum()
    #     progress_bar(batch_idx, len(testloader), 'Name: %s Loss: %.3f | Acc: %.3f%% (%d/%d)' % ("fuyic", test_loss/(batch_idx+1), 100. * correct/total, correct, total))
    #
    # for batch_idx, (feats0, feats1, feats2, targets, names) in enumerate(testloader_fzc):
    #     feats0, feats1, feats2, targets = feats0.cuda(), feats1.cuda(), feats2.cuda(), targets.cuda()
    #     outputs = net(feats0, feats1, feats2)
    #     loss = criterion(outputs, targets)
    #
    #     test_loss += loss.data
    #     _, predicted = torch.max(outputs.data, 1)
    #     total += targets.size(0)
    #     correct += predicted.eq(targets.data).cpu().sum()
    #     progress_bar(batch_idx, len(testloader), 'Name: %s Loss: %.3f | Acc: %.3f%% (%d/%d)' % ("fzc", test_loss/(batch_idx+1), 100. * correct/total, correct, total))
    #
    # for batch_idx, (feats0, feats1, feats2, targets, names) in enumerate(testloader_renji):
    #     feats0, feats1, feats2, targets = feats0.cuda(), feats1.cuda(), feats2.cuda(), targets.cuda()
    #     outputs = net(feats0, feats1, feats2)
    #     loss = criterion(outputs, targets)
    #
    #     test_loss += loss.data
    #     _, predicted = torch.max(outputs.data, 1)
    #     total += targets.size(0)
    #     correct += predicted.eq(targets.data).cpu().sum()
    #     progress_bar(batch_idx, len(testloader), 'Name: %s Loss: %.3f | Acc: %.3f%% (%d/%d)' % ("renji", test_loss/(batch_idx+1), 100. * correct/total, correct, total))


for epoch in range(100):
    train()
    test()
    torch.save(net.state_dict(), "checkpoints/" + str(epoch) + ".pt")
