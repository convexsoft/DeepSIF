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


    epsilon = 0.1
    theta11 = round(np.random.random(1)[0], 2)
    theta12 = round((1-theta11)*(1-np.random.random(1)[0]),2)
    theta21 = round(np.random.random(1)[0],2)
    theta22 = round((1 - theta21)*(1 - np.random.random(1)[0]),2)
    c1 = round(5*np.random.random(1)[0],2)
    c2 = round(5*np.random.random(1)[0],2)
    err = 10e-5
    err1 = 1

    logp1 = 0
    logp2 = 0
    while err1>err:
        tlogp1 = logp1
        tlogp2 = logp2
        # p1 = c1*p1**theta11*p2**theta12
        # p2 = c2*p1**theta21*p2**theta22
        logp1= np.log(c1)+theta11*logp1+theta12*logp2
        logp2 = np.log(c2) + theta21 * logp2 + theta22 * logp2
        err = np.abs(logp1-tlogp1)+np.abs(logp2-tlogp2)
    # if p1<1000 and p1>0.1:
        num = num +1
        theta1_data.append([float(theta11),float(theta12),float(np.log(c1))])
        theta2_data.append([float(theta21),float(theta22),float(np.log(c2))])
        y_data.append([float(logp1),float(logp2)])
    print("log class:", y_data)

    err2 = 1
    p1=0
    p2=0
    while err2 > err:
        tp1 = p1
        tp2 = p2
        p1 = c1 * p1 ** theta11 * p2 ** theta12
        p2 = c2 * p1 ** theta21 * p2 ** theta22
        err2 = np.abs(np.log(p1) - np.log(tp1)) + np.abs(np.log(p2) - np.log(tp2))


        num = num + 1
        theta1_data.append([float(theta11), float(theta12), float(np.log(c1))])
        theta2_data.append([float(theta21), float(theta22), float(np.log(c2))])
        y_data.append([float(np.log(p1)), float(np.log(p2))])
    print("==class:", y_data)


generate_data(1)
