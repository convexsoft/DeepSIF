from cvxpylayers.torch import CvxpyLayer

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from random import seed
from random import random


def transition_matrix(n):
    trans=np.random.random((n, n))
    for i in range(n):
        u=0
        for j in range(n):
            u=u+trans[i,j]
        for j in range(n):
            trans[i,j]=trans[i,j]/u
    return trans

def vector_v(n):
    v = np.random.random(n)
    u=0
    for j in range(n):
        u=u+v[j]
    for j in range(n):
        v[j]=v[j]/u
    return v


def pagerank(M, num_iterations: int = 100, d: float = 0.85):

    N = M.shape[1]
    v = vector_v(N)
    v = v / np.linalg.norm(v, 1)
    M_hat = (d * M + (1 - d) / N)
    for i in range(num_iterations):
        v = M_hat @ v
    return v

M=transition_matrix(10)

v = pagerank(np.transpose(M), 100, 0.85)


#Data generation

n = 2
m = 2

# Number of training input-output pairs
N = 100

# Number of validation pairs
N_val = 50

torch.random.manual_seed(243)
np.random.seed(243)

normal = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(n), torch.eye(n))
lognormal = lambda batch: torch.exp(normal.sample(torch.tensor([batch])))

A_true = torch.randn((m, n))
c_true = np.abs(torch.randn(m))


def generate_data(num_points, seed):
    inputs=[]
    outputs = []
    for i in range(num_points):
        A = np.random.random((n, m))
        P = np.random.random((n, m))
        I = np.identity(m)
        v = np.random.random(n)
        v[1]=1-v[0]
        for j in range(m):
            A[j,0]=np.random.random()
            A[j,1]=1-A[j,0]
        for k in range(1000):
            v=np.dot(np.transpose(A),v)
        inputs.append(A.flatten())
        outputs.append(v)
    return inputs, torch.stack([torch.tensor(t) for t in outputs])

train_inputs, train_outputs = generate_data(N, 243)

torch.random.manual_seed(243)
np.random.seed(243)

val_inputs, val_outputs = generate_data(N_val, 0)


#Monomial fit to each component

log_c = cp.Variable(shape=(m,1))
theta = cp.Variable(shape=(n*m,1))
inputs_np = np.array(train_inputs)
#inputs_np = train_inputs.numpy()
log_outputs_np = np.log(train_outputs.numpy()).T
log_inputs_np = np.log(inputs_np).T
offsets = cp.hstack([log_c]*N)

cp_preds = theta.T @ log_inputs_np + offsets
objective_fn = (1/N) * cp.sum_squares(cp_preds - log_outputs_np)
lstq_problem = cp.Problem(cp.Minimize(objective_fn))
lstq_problem.is_dcp()
lstq_problem.solve(verbose=True)

c = torch.exp(torch.tensor(log_c.value)).squeeze()
lstsq_val_preds = []
val_inputs=torch.tensor(val_inputs)
for i in range(N_val):
    inp = val_inputs[i, :].numpy()
    pred = cp.multiply(c,cp.gmatmul(theta.T.value, inp))
    lstsq_val_preds.append(pred.value)

cp_preds = theta.T @ log_inputs_np + offsets
objective_fn = (1/N) * cp.sum_squares(cp_preds - log_outputs_np)
lstq_problem = cp.Problem(cp.Minimize(objective_fn))

lstq_problem.solve(verbose=True)

c = torch.exp(torch.tensor(log_c.value)).squeeze()
lstsq_val_preds = []
for i in range(N_val):
    inp = val_inputs[i, :].numpy()
    pred = cp.multiply(c,cp.gmatmul(theta.T.value, inp))
    lstsq_val_preds.append(pred.value)

#Fitting
A_param = cp.Parameter(shape=(m,n))
c_param = cp.Parameter(pos=True, shape=(m,))
x_slack = cp.Variable(pos=True, shape=(n,))
x_param = cp.Parameter(pos=True, shape=(n,))
y = cp.Variable(pos=True, shape=(m,))

prediction = cp.multiply(c_param, cp.gmatmul(A_param, x_slack))
objective_fn = cp.sum(prediction / y + y / prediction)
# constraints = [x_slack == x_param]
# for i in range(m-1):
#     constraints += [y[i] <= y[i+1]]
problem = cp.Problem(cp.Minimize(objective_fn))
problem.is_dgp(dpp=True)
A_param.value = np.random.randn(m,n)
x_param.value = np.abs(np.random.randn(n))
c_param.value = np.abs(np.random.randn(m))

layer = CvxpyLayer(problem, parameters=[A_param, c_param, x_param], variables=[y], gp=True)
torch.random.manual_seed(1)
A_tch = torch.tensor(theta.T.value.reshape(2,2))
A_tch.requires_grad_(True)
c_tch = torch.tensor(np.squeeze(np.exp(log_c.value)))
c_tch.requires_grad_(True)
train_losses = []
val_losses = []

lam1 = torch.tensor(1e-1)
lam2 = torch.tensor(1e-1)

opt = torch.optim.SGD([A_tch, c_tch], lr=5e-2)
for epoch in range(10):
    preds = layer(A_tch, c_tch, torch.tensor(train_inputs).reshape(200,2), solver_args={'acceleration_lookback': 0})[0]
    loss = (preds - torch.cat((train_outputs,train_outputs))).pow(2).sum(axis=1).mean(axis=0)

    with torch.no_grad():
        val_preds = layer(A_tch, c_tch, val_inputs, solver_args={'acceleration_lookback': 0})[0]
        val_loss = (val_preds - val_outputs).pow(2).sum(axis=1).mean(axis=0)

    print('(epoch {0}) train / val ({1:.4f} / {2:.4f}) '.format(epoch, loss, val_loss))
    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    opt.zero_grad()
    loss.backward()
    opt.step()
    with torch.no_grad():
        c_tch = torch.max(c_tch, torch.tensor(1e-8))