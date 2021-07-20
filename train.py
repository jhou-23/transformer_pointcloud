from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm
from dataset import ShapeNetDataset, ModelNet40
from model import PointTransformer, PCT, PointTransformer2

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=1024, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--model_type', type=str, default='PointTransformer', help="which model")
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
parser.add_argument('--optimizer', type=str, default='adam', help="optimizer adam|sgd")
parser.add_argument('--logname', type=str, default='', help='logname')

opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)
print(opt.manualSeed)
random.seed(opt.manualSeed)

if opt.dataset_type == 'shapenet':
    train_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        npoints=opt.num_points)

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
elif opt.dataset_type == 'modelnet40':
    # train_dataset = ModelNet40(
    #     num_points = opt.num_points,
    #     partition='train')

    # test_dataset = ModelNet40(
    #     partition='test', num_points = opt.num_points)
    train_dataset = ModelNet40(root=opt.dataset, npoints=opt.num_points, split="train", data_augmentation=True)
    test_dataset = ModelNet40(root=opt.dataset, npoints=opt.num_points, split="test", data_augmentation=False)

else:
    exit('wrong dataset type')

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

print(len(train_dataset), len(test_dataset))
# num_classes = len(train_dataset.classes)
# print('classes', num_classes)

try:
    os.makedirs("models/"+opt.outf)
except OSError:
    pass

if opt.model_type == "PointTransformer":
    classifier = PointTransformer(num_heads=1, d_qk=64, d_v=64)
elif opt.model_type == "PCT":
    classifier = PCT()
else:
    print("Bad input")
    exit()

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

if opt.optimizer == "adam":
    optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
elif opt.optimizer == "sgd":
    optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch, eta_min=0.01)

criterion = nn.CrossEntropyLoss()
classifier.cuda()


num_batch = len(train_dataset) / opt.batchSize

fadksfadfad = open("logs/log-%s.txt" %opt.logname, 'w')
fadksfadfad.close()

with open("logs/log-%s.txt" %opt.logname, 'a') as f:
    f.write(str(opt))
    f.write('\n')
    f.write(str(classifier))
    f.write('\n')
    f.write('\n')

for epoch in range(opt.nepoch):
    
    epoch_train_accuracy = 0
    epoch_train_loss = 0
    counter = 0
    for i, data in enumerate(train_loader, 0):
        points, target = data
        target = target[:, 0]
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred = classifier(points)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))
        
        with open("logs/log-%s.txt" %opt.logname, 'a') as f:
            f.write('[%d: %d/%d] train loss: %f accuracy: %f\n' % (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))

        epoch_train_loss += loss.item()
        epoch_train_accuracy += correct.item()
        counter += 1
        # if i % 10 == 0:
        #     j, data = next(enumerate(test_loader, 0))
        #     points, target = data
        #     target = target[:, 0]
        #     points, target = points.cuda(), target.cuda()
        #     classifier = classifier.eval()
        #     pred = classifier(points)
        #     loss = F.nll_loss(pred, target)
        #     pred_choice = pred.data.max(1)[1]
        #     correct = pred_choice.eq(target.data).cpu().sum()
        #     print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, 'test', loss.item(), correct.item()/float(opt.batchSize)))
        #     with open("log-%s.txt" %opt.dataset_type, 'a') as f:
        #         f.write('[%d: %d/%d] %s loss: %f accuracy: %f\n' % (epoch, i, num_batch, 'test', loss.item(), correct.item()/float(opt.batchSize)))
    epoch_test_accuracy = 0
    epoch_test_loss = 0
    tcounter = 0
    for j, data in enumerate(test_loader, 0):
        points, target = data
        target = target[:, 0]
        points, target = points.cuda(), target.cuda()
        classifier = classifier.eval()
        pred = classifier(points)
        loss = criterion(pred, target)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        epoch_test_loss += loss.item()
        epoch_test_accuracy += correct.item()
        tcounter += 1

    epoch_test_accuracy = epoch_test_accuracy / (tcounter * float(opt.batchSize))
    epoch_test_loss = epoch_test_loss / tcounter

    print("-" * 40)
    print("EPOCH %d stats" % epoch)
    print('[%d] %s loss: %f accuracy: %f' % (epoch, 'test', epoch_test_loss, epoch_test_accuracy))
    with open("logs/log-%s.txt" %opt.logname, 'a') as f:
        f.write("-" * 40 + "\n")
        f.write('[%d] %s loss: %f accuracy: %f\n' % (epoch, 'test', epoch_test_loss, epoch_test_accuracy))
    
    epoch_train_loss = epoch_train_loss / counter
    epoch_train_accuracy = epoch_train_accuracy / (counter * float(opt.batchSize))
    print('[%d] %s loss: %f accuracy: %f' % (epoch, 'train', epoch_train_loss, epoch_train_accuracy))
    with open("logs/log-%s.txt" %opt.logname, 'a') as f:
        f.write('[%d] %s loss: %f accuracy: %f\n' % (epoch, 'train', epoch_train_loss, epoch_train_accuracy))
        f.write("-" * 40 + "\n")
    print("-" * 40)

    scheduler.step()
    torch.save(classifier.state_dict(), 'models/%s/cls_model_%d.pth' % (opt.outf, epoch))

total_correct = 0
total_testset = 0
for i,data in tqdm(enumerate(test_loader, 0)):
    points, target = data
    target = target[:, 0]
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

print("final accuracy {}".format(total_correct / float(total_testset)))
with open("logs/log-%s.txt" %opt.logname, 'a') as f:
    f.write("final accuracy {}\n".format(total_correct / float(total_testset)))