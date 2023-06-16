from torch.autograd import Variable
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.datasets import fetch_california_housing
import torch
import torch.nn as nn
from torch.optim import SGD


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
        while err>10e-5:
            tp1 = p1
            tp2 = p2
            p1 = c1*p1**theta11*p2**theta12
            p2 = c2*p1**theta21*p2**theta22
            err = np.abs(p1-tp1)+np.abs(p2-tp2)
        if p1<1000 and p1>0.1:
            # print("p1:",p1)
            # print("p2:", p2)
            # print(theta11,theta12,theta21,theta22)
            num = num +1
            theta1_data.append([float(theta11),float(theta12),float(np.log(c1))])
            theta2_data.append([float(theta21),float(theta22),float(np.log(c2))])

            y_data.append([float(np.log(p1)),float(np.log(p2))])
    return theta1_data, theta2_data, y_data




class MLPPregression(nn.Module):
    def __init__(self):
        super(MLPPregression, self).__init__()
        self.hidden1_1 = nn.Linear(in_features=3,out_features=1,bias=False)
        self.hidden2_1 = nn.Linear(3,1,bias=False)
        self.hidden1_2 = nn.Linear(3,1,bias=False)
        self.hidden2_2 = nn.Linear(3,1,bias=False)
        self.hidden1_3 = nn.Linear(3,1,bias=False)
        self.hidden2_3 = nn.Linear(3,1,bias=False)

        self.hidden1_4 = nn.Linear(3, 1, bias=False)
        self.hidden2_4 = nn.Linear(3, 1, bias=False)
        self.hidden1_5 = nn.Linear(3, 1, bias=False)
        self.hidden2_5 = nn.Linear(3, 1, bias=False)

        self.hidden1_f = nn.Linear(3,1, bias=False)
        self.hidden2_f = nn.Linear(3,1, bias=False)

    def forward(self,x1,x2):
        batch_size = len(x1)
        b = torch.tensor([[1]*batch_size]).T
        l1 = F.relu6(self.hidden1_1(x1))
        l2 = F.relu6(self.hidden2_1(x2))

        l_out = torch.cat((l1, l2, b), 1)
        l1 = F.relu6(self.hidden1_2(x1*l_out))
        l2 = F.relu6(self.hidden2_2(x2*l_out))

        l_out = torch.cat((l1, l2, b), 1)
        l1 = F.relu(self.hidden1_3(x1 * l_out))
        l2 = F.relu(self.hidden2_3(x2 * l_out))

        l_out = torch.cat((l1, l2, b), 1)
        l1 = F.relu(self.hidden1_4(x1 * l_out))
        l2 = F.relu(self.hidden2_4(x2 * l_out))

        l_out = torch.cat((l1, l2, b), 1)
        l1 = F.relu(self.hidden1_5(x1 * l_out))
        l2 = F.relu(self.hidden2_5(x2 * l_out))

        l_out = torch.cat((l1, l2, b), 1)
        l1 = self.hidden1_f(x1 * l_out)
        l2 = self.hidden2_f(x2 * l_out)

        output = torch.cat((l1,l2), 1)
        # output = torch.sigmoid(self.predict(x))
        return output


def custom_mse(predicted, target):
    total_mse = 0

    for i in range(target.shape[1]):
        # print("predicted[i]:", predicted.T[i])
        total_mse+=nn.MSELoss()(predicted.T[i], target.T[i])
    return total_mse


def training_progress(train_loader):
    epoch_num = 300
    mlpreg = MLPPregression()
    optimizer = SGD(mlpreg.parameters(),lr=10e-4,weight_decay=0.0001)
    train_loss_all = []

    theta1_data_test, theta2_data_test, y_data_test = generate_data(1)
    y_pred1 = []
    y_true1 = [y_data_test[0][0]]*epoch_num
    # print("y_true:", y_data_test[0])
    print("y_true1:", y_true1)

    y_pred2 = []
    y_true2 = [y_data_test[0][1]]*epoch_num
    print("y_true1:", y_true1)

    test_x1t = torch.tensor(theta1_data_test)
    test_x2t = torch.tensor(theta2_data_test)
    test_yt = torch.tensor(y_data_test)


    for epoch in range(epoch_num):
        print("epoch:", epoch)

        train_loss = 0
        train_num = 0
        for step,(b_x1,b_x2,b_y) in enumerate(train_loader):
            # print("step:", step)
            output = mlpreg(b_x1,b_x2)

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

            train_loss += loss.item() * b_x1.size(0)
            train_num += train_loss
        print("train_loss:", train_loss / train_num)

        train_loss_all.append(train_loss / train_num)
        # single_predict = mlpreg(s_feat_st)
        single_predict = mlpreg(test_x1t,test_x2t)
        p = single_predict.data.numpy()
        # print("p:", p)
        y_pred1.append(p[0][0])
        y_pred2.append(p[0][1])

        # print("y_pred:", y_pred)

    plt.plot(y_pred1, label='y1_pred', c="g", marker="o", linewidth=2)
    plt.plot(y_true1, label='y1_true', c="red", marker="o", linewidth=2)

    plt.plot(y_pred2, label='y2_pred', c="blue", linewidth=2)
    plt.plot(y_true2, label='y2_true', c="red", linewidth=2)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    theta1_data, theta2_data, y_data = generate_data(3000)
    print("len:",len(theta1_data))
    train_x1t = torch.tensor(theta1_data)
    train_x2t = torch.tensor(theta2_data)
    train_yt = torch.tensor(y_data)

    train_data = Data.TensorDataset(train_x1t, train_x2t, train_yt)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=50, shuffle=True, num_workers=0)

    training_progress(train_loader)


