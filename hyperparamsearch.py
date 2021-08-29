from __future__ import print_function
import argparse
import sys
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
from model import PointTransformer, PCT
import yaml
from pprint import pprint

test_rate = 1
cfname = sys.argv[1]
with open("configs/" + cfname) as f:
    config = yaml.safe_load(f)

pprint(config)
config["manualSeed"] = random.randint(1, 10000)
print(config["manualSeed"])
random.seed(config["manualSeed"])

if config["dataset"]["type"] == 'shapenet':
    train_dataset = ShapeNetDataset(
        root=config["dataset"]["path"],
        classification=True,
        npoints=config["dataset"]["num_points"])
    test_dataset = ShapeNetDataset(
        root=["dataset"]["path"],
        classification=True,
        split='test',
        npoints=config["dataset"]["num_points"],
        data_augmentation=False)
elif config["dataset"]["type"] == 'modelnet40':
    train_dataset = ModelNet40(
        root=config["dataset"]["path"],
        npoints=config["dataset"]["num_points"],
        bins = config["dataset"]["bins"],
        split="train", 
        data_augmentation=True)
    test_dataset = ModelNet40(root=config["dataset"]["path"],
        npoints=config["dataset"]["num_points"],
        bins = config["dataset"]["bins"],
        split="test",
        data_augmentation=False)
else:
    exit('wrong dataset type')

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=config["train"]["batch_size"],
    shuffle=True,
    num_workers=config["train"]["workers"])

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=config["train"]["batch_size"],
    shuffle=True,
    num_workers=config["train"]["workers"])

print(len(train_dataset), len(test_dataset))



num_batch = len(train_dataset) / config["train"]["batch_size"]

sigma_choices = [1.0, 0.3, 0.03, 0.003]
performances = dict(zip(sigma_choices, [0] * len(sigma_choices)))
og_experiment_name = config["experiment_name"]
for sigma in sigma_choices:

    print("sigma: " + str(sigma))
    config["experiment_name"] = og_experiment_name + "-sigma" + str(sigma)
    config["sigma"] = sigma
    try:
        os.makedirs("models/"+config["experiment_name"])
    except OSError:
        pass

    if config["model"]["type"] == "PointTransformer":
        classifier = PointTransformer(config)
    elif config["model"]["type"] == "PCT":
        classifier = PCT(config)
    else:
        print("Bad Model")
        exit()

    if config["model"]["pretrained"] != None:
        classifier.load_state_dict(torch.load(config["model"]["pretrained"]))

    if config["optimizer"]["type"] == "Adam":
        if "eps" in config["optimizer"].keys():
            optimizer = optim.Adam(classifier.parameters(), lr=config["optimizer"]["lr"], betas=config["optimizer"]["betas"], eps=config["optimizer"]["eps"], weight_decay=float(config["optimizer"]["weight_decay"]))
        else:
            optimizer = optim.Adam(classifier.parameters(), lr=config["optimizer"]["lr"], betas=config["optimizer"]["betas"], weight_decay=float(config["optimizer"]["weight_decay"]))
    elif config["optimizer"]["type"] == "SGD":
        optimizer = optim.SGD(classifier.parameters(), lr=config["optimizer"]["lr"], momentum=config["optimizer"]["momentum"], weight_decay=float(config["optimizer"]["weight_decay"]))

    if config["scheduler"]["type"] == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config["train"]["epochs"], eta_min=config["scheduler"]["eta_min"])
    elif config["scheduler"]["type"] == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["scheduler"]["step_size"], gamma=config["scheduler"]["gamma"])
    elif config["scheduler"]["type"] == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                    verbose=True,
                                                    factor=config["scheduler"]["factor"],
                                                    patience=config["scheduler"]["patience"])
    else:
        print("Bad Scheduler")
        exit()

    criterion = nn.CrossEntropyLoss()
    classifier.cuda()
    if config["save_data"]["save_log"]:
        lfile = open("logs/log-%s.txt" % config["experiment_name"], 'w')
        lfile.close()

        with open("logs/log-%s.txt" % config["experiment_name"], 'a') as f:
            f.write(str(config))
            f.write('\n')
            f.write(str(classifier))
            f.write('\n')
            f.write('\n')
    for epoch in range(config["train"]["epochs"]):
        epoch_train_accuracy = 0
        epoch_train_loss = 0
        counter = 0
        for i, data in enumerate(train_loader, 0):
            points, discrete_points, target = data
            target = target[:, 0]
            points, discrete_points, target = points.cuda(), discrete_points.cuda(), target.cuda()
            optimizer.zero_grad()
            classifier = classifier.train()
            pred = classifier(points, discrete_points)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            if config["model"]["components"]["input_embedding"]["pos_enc"] == "grpe":
                with torch.no_grad():
                    sigmas = []
                    for j in range(len(classifier.sa)):
                        # ensure it does not go beyond (0,inf)
                        sigmas.append(torch.clamp(classifier.sa[j].pos_encoder.sigma, min=1e-3, max=100).item())
                        clamped_value = nn.Parameter(torch.clamp(classifier.sa[j].pos_encoder.sigma, min=1e-3, max=100))
                    # if epoch % test_rate == 0 and i==len(train_loader)-1:
                    print("sigmas" + str(sigmas))
                    if config["save_data"]["save_log"]:
                        with open("logs/log-%s.txt" % config["experiment_name"], 'a') as f:
                            f.write("sigmas" + str(sigmas) + "\n")

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(config["train"]["batch_size"])))
            
            if config["save_data"]["save_log"]:
                with open("logs/log-%s.txt" % config["experiment_name"], 'a') as f:
                    f.write('[%d: %d/%d] train loss: %f accuracy: %f\n' % (epoch, i, num_batch, loss.item(), correct.item() / float(config["train"]["batch_size"])))

            epoch_train_loss += loss.item()
            epoch_train_accuracy += correct.item()
            counter += 1

        epoch_test_accuracy = 0
        epoch_test_loss = 0
        tcounter = 0
        print("EPOCH %d stats" % epoch)
        if config["save_data"]["save_log"]:
            with open("logs/log-%s.txt" % config["experiment_name"], 'a') as f:
                f.write("EPOCH %d stats\n" % epoch)

        if epoch % test_rate == 0:
            with torch.no_grad():
                for j, data in enumerate(test_loader, 0):
                    points, discrete_points, target = data
                    target = target[:, 0]
                    points, discrete_points, target = points.cuda(), discrete_points.cuda(), target.cuda()
                    classifier = classifier.eval()
                    pred = classifier(points, discrete_points)
                    loss = criterion(pred, target)
                    pred_choice = pred.data.max(1)[1]
                    correct = pred_choice.eq(target.data).cpu().sum()
                    epoch_test_loss += loss.item()
                    epoch_test_accuracy += correct.item()
                    tcounter += 1

            epoch_test_accuracy = epoch_test_accuracy / (tcounter * float(config["train"]["batch_size"]))
            epoch_test_loss = epoch_test_loss / tcounter
            
            print("-" * 40)
            print('[%d] %s loss: %f accuracy: %f' % (epoch, 'test', epoch_test_loss, epoch_test_accuracy))
            if config["save_data"]["save_log"]:
                with open("logs/log-%s.txt" % config["experiment_name"], 'a') as f:
                    f.write("-" * 40 + "\n")
                    f.write('[%d] %s loss: %f accuracy: %f\n' % (epoch, 'test', epoch_test_loss, epoch_test_accuracy))
                    f.write("-" * 40 + "\n")
            print("-" * 40)
        
        print("-" * 40)
        epoch_train_loss = epoch_train_loss / counter
        epoch_train_accuracy = epoch_train_accuracy / (counter * float(config["train"]["batch_size"]))
        print('[%d] %s loss: %f accuracy: %f' % (epoch, 'train', epoch_train_loss, epoch_train_accuracy))
        if config["save_data"]["save_log"]:    
            with open("logs/log-%s.txt" % config["experiment_name"], 'a') as f:
                f.write("-" * 40 + "\n")
                f.write('[%d] %s loss: %f accuracy: %f\n' % (epoch, 'train', epoch_train_loss, epoch_train_accuracy))
                f.write("-" * 40 + "\n")
        print("-" * 40)

        scheduler.step()
        if config["save_data"]["save_model"]:
            torch.save(classifier.state_dict(), 'models/%s/cls_model_%d.pth' % (config["experiment_name"], epoch))

    total_correct = 0
    total_testset = 0
    with torch.no_grad():
        for i,data in tqdm(enumerate(test_loader, 0)):
            points, discrete_points, target = data
            target = target[:, 0]
            points, discrete_points, target = points.cuda(), discrete_points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred = classifier(points, discrete_points)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            total_correct += correct.item()
            total_testset += points.size()[0]

    print("final accuracy {}".format(total_correct / float(total_testset)))
    performances[sigma] = total_correct / float(total_testset)
    if config["save_data"]["save_log"]:
        with open("logs/log-%s.txt" %config["experiment_name"], 'a') as f:
            f.write("final accuracy {}\n".format(total_correct / float(total_testset)))

print(performances)