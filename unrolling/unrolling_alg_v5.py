import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data as Data
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
        c1 = round(5*np.random.random(1)[0],2)
        c2 = round(5*np.random.random(1)[0],2)
        err = 1
        p1 = 1
        p2 = 1
        while err>10e-7:
            tp1 = p1
            tp2 = p2
            p1 = c1*p1**theta11*p2**theta12
            p2 = c2*p1**theta21*p2**theta22
            err = np.abs(p1-tp1)+np.abs(p2-tp2)
        if p1 < 10e3 and p1 > 10e-1:
            # print("p1:",p1)
            # print("p2:", p2)
            # print(theta11,theta12,theta21,theta22)
            num = num +1
            theta1_data.append([float(theta11),float(theta12),float(np.log(c1))])
            theta2_data.append([float(theta21),float(theta22),float(np.log(c2))])

            y_data.append([float(np.log(p1)),float(np.log(p2))])
    # print("theta1_data:", theta1_data)
    # print("theta2_data:", theta2_data)
    # print("y_data:", y_data)
    # return torch.tensor(theta1_data), torch.tensor(theta2_data), torch.tensor(y_data)
    return theta1_data, theta2_data, y_data


class MLPPregression(nn.Module):
    def __init__(self):
        super(MLPPregression, self).__init__()
        self.hidden1 = nn.Linear(in_features=6,out_features=100,bias=True)
        self.hidden2 = nn.Linear(100,100)
        self.hidden3 = nn.Linear(100, 100)
        self.hidden4 = nn.Linear(100,50)
        # self.hidden5 = nn.Linear(50, 50)
        self.predict = nn.Linear(50,2)
    def forward(self,x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        # x = F.relu(self.hidden5(x))
        output = self.predict(x)
        # output = torch.sigmoid(self.predict(x))
        return output


def custom_mse(predicted, target):
    total_mse = 0
    # print("predicted:", predicted)
    # print("target:", target)

    for i in range(target.shape[1]):
        # print("predicted[i]:", predicted.T[i])
        total_mse+=nn.MSELoss()(predicted.T[i], target.T[i])
    return total_mse


def training_progress(train_loader):
    epoch_num = 500
    mlpreg = MLPPregression()
    optimizer = SGD(mlpreg.parameters(),lr=1*10e-4,weight_decay=10e-4)
    train_loss_all = []

    theta1_data_test, theta2_data_test, y_data_test = data_read(path="./keep_data/test_data.csv")
    # theta1_data_test, theta2_data_test, y_data_test = generate_data(1)
    y_pred1 = []
    y_true1 = [y_data_test[0][0]]*epoch_num
    # print("y_true:", y_data_test[0])
    print("y_true1:", y_true1)

    y_pred2 = []
    y_true2 = [y_data_test[0][1]]*epoch_num
    print("y_true1:", y_true1)

    test_xt = torch.cat((torch.tensor(theta1_data_test), torch.tensor(theta2_data_test)), 1)
    test_yt = torch.tensor(y_data_test)


    for epoch in range(epoch_num):
        print("epoch:", epoch)

        train_loss = 0
        train_num = 0
        for step,(b_x,b_y) in enumerate(train_loader):
            # print("step:", step)
            output = mlpreg(b_x)

            # print("output:", output)
            # loss = loss_func(output,b_y)
            loss = custom_mse(output, b_y)
            # print("loss:", loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * b_x.size(0)
            train_num += b_x.size(0)
        print("train_loss:", train_loss / train_num)
        train_loss_all.append(train_loss / train_num)
        # single_predict = mlpreg(s_feat_st)
        single_predict = mlpreg(test_xt)
        p = single_predict.data.numpy()
        # print("p:", p)
        y_pred1.append(p[0][0])
        y_pred2.append(p[0][1])

        # print("y_pred:", y_pred)

    plt.plot(y_pred1, label='x1_pred', c="#77AC30", linewidth=1)
    plt.plot(y_true1, label='x1_true', c="#D85319", linewidth=2)

    plt.plot(y_pred2, label='x2_pred', c="#0072BD", linewidth=1)
    plt.plot(y_true2, label='x2_true', c="#7E2F8E", linewidth=2)
    plt.xlabel(r'$Epoch$')
    plt.ylabel(r'$Dual power$')
    plt.legend()
    plt.savefig("unrolling_v5_1.pdf")
    plt.show()


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


def data_read(path = "./keep_data/training_data.csv"):
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
    theta1_data, theta2_data, y_data = generate_data(2000)
    # theta1_data, theta2_data, y_data = data_read()
    print("len:",len(theta1_data))
    # x_train_s = theta1_data, theta2_data
    train_xt = torch.cat((torch.tensor(theta1_data),torch.tensor(theta2_data)),1)
    train_yt = torch.tensor(y_data)

    train_data = Data.TensorDataset(train_xt, train_yt)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=50, shuffle=True, num_workers=0)

    training_progress(train_loader)


