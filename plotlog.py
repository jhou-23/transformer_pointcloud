import matplotlib.pyplot as plt
import numpy as np
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument("--logname", type=int, required=True)
parser.add_argument("--ftype", type=str, required=True, help="old | new")
parser.add_argument("--plot", type=str2bool, required=True)
opt = parser.parse_args()

f = open("log-" + opt.logname + ".txt", 'r')

train_i = []
test_i = []

train_acc = []
train_loss = []

test_acc = []
test_loss = []

sample = 1
train_counter = 0
test_counter = 0
for line in f:
    line.strip().strip()
    if opt.ftype == "old":
        if "train" in line:
            if train_counter % sample == 0:
                data = line.split(": ")
                data[0] = int(data[0][1:])
                num, den = data[1][:-12].split("/")
                num, den = float(num), float(den)
                data[1] = num / den
                data[2] = float(data[2][:-9])
                data[3] = float(data[3][:-1])
                data[0] += data.pop(1)
                train_i.append(data[0])
                train_loss.append(data[1])
                train_acc.append(data[2])
            train_counter += 1
        elif "test" in line:
            if test_counter % sample == 0:
                data = line.split(": ")
                data[0] = int(data[0][1:])
                num, den = data[1][:-11].split("/")
                num, den = float(num), float(den)
                data[1] = num / den
                data[2] = float(data[2][:-9])
                data[3] = float(data[3][:-1])
                data[0] += data.pop(1)
                test_i.append(data[0])
                test_loss.append(data[1])
                test_acc.append(data[2])
    else:
        # if "train" in line:
        #     if train_counter % sample == 0:
        #         data = line.split(": ")
        #         data[0] = int(data[0][1:])
        #         num, den = data[1][:-12].split("/")
        #         num, den = float(num), float(den)
        #         data[1] = num / den
        #         data[2] = float(data[2][:-9])
        #         data[3] = float(data[3][:-1])
        #         data[0] += data.pop(1)
        #         train_i.append(data[0])
        #         train_loss.append(data[1])
        #         train_acc.append(data[2])
        #     train_counter += 1
        # elif "test" in line:
        #     if test_counter % sample == 0:
        #         data = line.split(": ")
        #         data[0] = int(data[0][1:])
        #         num, den = data[1][:-11].split("/")
        #         num, den = float(num), float(den)
        #         data[1] = num / den
        #         data[2] = float(data[2][:-9])
        #         data[3] = float(data[3][:-1])
        #         data[0] += data.pop(1)
        #         test_i.append(data[0])
        #         test_loss.append(data[1])
        #         test_acc.append(data[2])
        print("Not done")
        exit()
f.close()


ltest = -1 * (int((len(test_acc) / 250 * 10)))
ltrain = -1 * int((len(train_acc) / 250 * 10))

if opt.plot:
    fig = plt.figure()
    plt.title(opt.logname + " accuracy")
    line1 = plt.plot(train_i, train_acc, label="train")
    line2 = plt.plot(test_i, test_acc, label="test")
    fig.legend()

    plt.figure()
    plt.title(opt.logname + " loss")
    line3 = plt.plot(train_i, train_loss, label="train")
    line4 = plt.plot(test_i, test_loss, label="test")
    plt.legend()

    plt.figure()
    plt.title(opt.logname + " acc -10")
    line3 = plt.plot(train_i[ltrain:], train_acc[ltrain:], label="train")
    line4 = plt.plot(test_i[ltest:], test_acc[ltest:], label="test")
    plt.legend()

    plt.show()
else:
    a = np.array(test_acc[ltest:])
    print(np.mean(a))