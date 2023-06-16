from torch.autograd import Variable
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.optim import SGD
import csv


def generate_data(n):
    num = 0
    theta1_data = []
    theta2_data = []
    y_data = []
    while num<n:
        epsilon = 0.1
        theta11 = round(np.random.random(1)[0], 2)
        theta12 = round((1-theta11)*(1-np.random.random(1)[0]),2)
        theta21 = round(np.random.random(1)[0],2)
        theta22 = round((1 - theta21)*(1 - np.random.random(1)[0]),2)
        c1 = round(10*np.random.random(1)[0],2)
        c2 = round(10*np.random.random(1)[0],2)
        err = 1
        p1 = 1
        p2 = 1
        while err>10e-5:
            tp1 = p1
            tp2 = p2
            p1 = c1*p1**theta11*p2**theta12
            p2 = c2*p1**theta21*p2**theta22
            err = np.abs(np.log(p1) - np.log(tp1)) + np.abs(np.log(p2) - np.log(tp2))
        if p1<10e2 and p1>10e-2:
            num = num +1
            theta1_data.append([float(theta11),float(theta12),float(np.log(c1))])
            theta2_data.append([float(theta21),float(theta22),float(np.log(c2))])

            y_data.append([float(np.log(p1)),float(np.log(p2))])
    return theta1_data, theta2_data, y_data


class MyLinear1(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1_1 = nn.Linear(in_features=3, out_features=100, bias=True)
        self.hidden2_1 = nn.Linear(3, 100)
        self.hidden1_2 = nn.Linear(100, 1)
        self.hidden2_2 = nn.Linear(100, 1)

    def forward(self, x):
        x1 = x[:,0:3]
        x2 = x[:,3:6]
        batch_size = len(x1)
        b = torch.tensor([[1] * batch_size]).T
        l1_1 = F.relu6(self.hidden1_1(x1))
        l2_1 = F.relu6(self.hidden2_1(x2))
        l1_2 = F.relu6(self.hidden1_2(l1_1))
        l2_2 = F.relu6(self.hidden2_2(l2_1))
        l_out = torch.cat((l1_2, l2_2, b), 1)
        x1_2_l_out = torch.cat((x, l_out), 1)
        return x1_2_l_out


class MyLinear2(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1_1 = nn.Linear(in_features=3, out_features=100, bias=True)
        self.hidden2_1 = nn.Linear(3, 100)
        self.hidden1_2 = nn.Linear(100, 1)
        self.hidden2_2 = nn.Linear(100, 1)

    def forward(self, x1_2_l_out):
        x1 = x1_2_l_out[:, 0:3]
        x2 = x1_2_l_out[:, 3:6]
        l_out = x1_2_l_out[:, 6:9]
        batch_size = len(x1)
        b = torch.tensor([[1] * batch_size]).T

        l1_1 = F.relu6(self.hidden1_1(x1*l_out))
        l2_1 = F.relu6(self.hidden2_1(x2*l_out))
        # l1_1 = F.relu6(self.hidden1_1( l_out))
        # l2_1 = F.relu6(self.hidden2_1( l_out))
        l1_2 = F.relu6(self.hidden1_2(l1_1))
        l2_2 = F.relu6(self.hidden2_2(l2_1))
        l_out = torch.cat((l1_2, l2_2, b), 1)
        x1_2_l_out = torch.cat((x1,x2, l_out), 1)

        return x1_2_l_out


class MyLinear3(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1_1 = nn.Linear(in_features=3, out_features=100, bias=True)
        self.hidden2_1 = nn.Linear(3, 100)
        self.hidden1_2 = nn.Linear(100, 1)
        self.hidden2_2 = nn.Linear(100, 1)

    def forward(self, x1_2_l_out):
        x1 = x1_2_l_out[:, 0:3]
        x2 = x1_2_l_out[:, 3:6]
        l_out = x1_2_l_out[:, 6:9]
        l1_1 = F.relu6(self.hidden1_1(x1 * l_out))
        l2_1 = F.relu6(self.hidden2_1(x2 * l_out))
        # l1_1 = F.relu6(self.hidden1_1(l_out))
        # l2_1 = F.relu6(self.hidden2_1(l_out))
        l1_2 = self.hidden1_2(l1_1)
        l2_2 = self.hidden2_2(l2_1)
        l_out = torch.cat((l1_2, l2_2), 1)
        return l_out


def training_pro2(train_loader):
    net = []
    layer_num = 1

    for i in range(layer_num):

        net.append(MyLinear2())

    my_model = nn.Sequential(MyLinear1( ), *net, MyLinear3( ))
    learning_rate = 10e-4
    optimizer = SGD(my_model.parameters(), lr=learning_rate, weight_decay=10e-4)
    epoch_num = 100


    # test_data
    # theta1_data_test, theta2_data_test, y_data_test = data_read(path = "test_data.csv")
    theta1_data_test, theta2_data_test, y_data_test = generate_data(1)
    y_pred1 = []
    y_true1 = [ y_data_test[0][0]] * epoch_num
    # print("y_true:", y_data_test[0])
    print("y_true1:", y_true1)

    y_pred2 = []
    y_true2 = [y_data_test[0][1]] * epoch_num
    print("y_true2:", y_true2)

    test_x1t = torch.tensor(theta1_data_test)
    test_x2t = torch.tensor(theta2_data_test)


    for epoch in range(epoch_num):
        print("epoch:", epoch)

        train_loss = 0
        train_num = 0
        for step,(b_x1,b_x2,b_y) in enumerate(train_loader):
            # print("step:", step)
            x = torch.cat((b_x1,b_x2),1)
            output = my_model(x)

            # print("output:", output)
            # print("b_y:", b_y)

            # loss = loss_func(output,b_y)
            loss = custom_mse(output, b_y)
            # print("loss:", loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print("loss:", loss)
            # print("loss.item:", loss.item())

            train_loss += loss.item()
            train_num += train_loss
        print("train_loss:", train_loss / train_num)

        # single_predict = mlpreg(s_feat_st)
        test_x = torch.cat((test_x1t, test_x2t), 1)
        single_predict = my_model(test_x)
        p = single_predict.data.numpy()
        # print("p:", p)
        y_pred1.append( p[0][0])
        y_pred2.append( p[0][1])

        # print("y_pred:", y_pred)
    plt.plot(y_pred1, label='y1_pred', c="g", marker="o", linewidth=2)
    plt.plot(y_true1, label='y1_true', c="red", marker="o", linewidth=2)

    plt.plot(y_pred2, label='y2_pred', c="blue", linewidth=2)
    plt.plot(y_true2, label='y2_true', c="red", linewidth=2)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.legend()
    plt.show()


def custom_mse(predicted, target):
    total_mse = 0
    # print("predicted:", predicted)
    for i in range(target.shape[1]):
        # print("predicted[i]:", predicted.T[i])
        total_mse+=nn.MSELoss()(predicted.T[i], target.T[i])
    return total_mse


def training_data_generate():
    data_num = 1000
    theta1_data, theta2_data, y_data = generate_data(data_num)
    print("len:", len(theta1_data))
    train_x1t = torch.tensor(theta1_data)
    train_x2t = torch.tensor(theta2_data)
    train_yt = torch.tensor(y_data)

    with open("training_data.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        for i in range(data_num):
            writer.writerow(theta1_data[i] + theta2_data[i] + y_data[i])
    # print("theta1_data:",theta1_data)
    return train_x1t, train_x2t, train_yt


def test_data_generate():
    data_num = 1
    theta1_data, theta2_data, y_data = generate_data(data_num)
    print("len:", len(theta1_data))
    train_x1t = torch.tensor(theta1_data)
    train_x2t = torch.tensor(theta2_data)
    train_yt = torch.tensor(y_data)

    with open("test_data.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        for i in range(data_num):
            writer.writerow(theta1_data[i] + theta2_data[i] + y_data[i])
    # print("theta1_data:",theta1_data)
    return train_x1t, train_x2t, train_yt


def data_read(path = "training_data.csv"):
    csvFile = open(path, "r")
    reader = csv.reader(csvFile)
    train_x1t = []
    train_x2t = []
    train_yt = []

    for item in reader:
        train_x1t.append([float(x) for x in item[:3]])
        train_x2t.append([float(x) for x in item[3:6]])
        train_yt.append([float(x) for x in item[-2:]])
    csvFile.close()
    print("train_x1t:", train_x1t)
    print("train_x2t:", train_x2t)

    print("train_yt:", train_yt)

    return torch.tensor(train_x1t), torch.tensor(train_x2t), torch.tensor(train_yt)



if __name__ == '__main__':
    data_num = 1000
    theta1_data, theta2_data, y_data = generate_data(data_num)
    print("len:", len(theta1_data))
    train_x1t = torch.tensor(theta1_data)
    train_x2t = torch.tensor(theta2_data)
    train_yt = torch.tensor(y_data)

    # train_x1t, train_x2t, train_yt = data_read()

    train_data = Data.TensorDataset(train_x1t, train_x2t, train_yt)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=50, shuffle=True, num_workers=0)
    #
    # training_progress(train_loader)

    training_pro2(train_loader)


