from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh
torch.set_default_tensor_type(torch.DoubleTensor)
torch.random.manual_seed(243)
np.random.seed(243)

def transition_matrix(n):
    trans=np.random.random((n, n))
    for i in range(n):
        u=0
        for j in range(n):
            u=u+trans[i,j]
        for j in range(n):
            trans[i,j]=trans[i,j]/u
    return trans


def generate_data(num_points, seed):
    inputs = 1*np.random.rand(num_points, n)
    # F = np.array([[0.8,0.2],[0.3,0.7]])
    F = transition_matrix(n)
    # F = np.array([[0.7,0.2,0.1],[0.12,0.75,0.23],[0.05,0.15,0.8]])
    outputs = 0.9*np.dot(F, inputs.T)+0.1*np.array([[1,1]]).T/n
    inputs_torch = torch.from_numpy(inputs)
    outputs_torch = torch.from_numpy(outputs.T)
    return inputs_torch, outputs_torch

#m,n**2
def monomial_fitting_lstsq(m,n,N,N_val,train_inputs,train_outputs,val_inputs):
    log_c = cp.Variable(shape=(m,1))
    theta = cp.Variable(shape=(n, m))
    inputs_np = train_inputs.numpy()
    log_outputs_np = np.log(train_outputs.numpy()).T
    log_inputs_np = np.log(inputs_np).T
    offsets = cp.hstack([log_c]*N)

    cp_preds = theta.T @ log_inputs_np + offsets
    objective_fn = (1/N) * cp.sum_squares(cp_preds - log_outputs_np)
    # print("objective_fn:", objective_fn.log_log_curvature)

    lstq_problem = cp.Problem(cp.Minimize(objective_fn))
    lstq_problem.is_dcp()
    lstq_problem.solve(verbose=True)

    c = torch.exp(torch.tensor(log_c.value)).squeeze()
    lstsq_val_preds = []
    for i in range(N_val):
        inp = val_inputs[i, :].numpy()
        pred = cp.multiply(c,cp.gmatmul(theta.T.value, inp))
        lstsq_val_preds.append(pred.value)
    return lstsq_val_preds, theta.value, log_c.value


def monomial_fitting_llcp(m,n,train_inputs,train_outputs,val_inputs,init_A, init_c):
    A_param = cp.Parameter(shape=(m, n))
    c_param = cp.Parameter(pos=True, shape=(m,))
    x_slack = cp.Variable(pos=True, shape=(n,))
    x_param = cp.Parameter(pos=True, shape=(n,))
    y = cp.Variable(pos=True, shape=(m,))
    punish = 10e3

    prediction = cp.multiply(c_param, cp.gmatmul(A_param, x_slack))

    objective_fn = cp.sum(prediction / y + y / prediction)

    print(objective_fn.log_log_curvature)

    constraints = [x_slack == x_param]
    # constraints = []
    # for i in range(m-1):
    #     constraints += [y[i] <= y[i+1]]
    problem = cp.Problem(cp.Minimize(objective_fn), constraints)
    # problem = cp.Problem(cp.Minimize(objective_fn))

    problem.is_dgp(dpp=True)

    A_param.value = np.random.randn(m, n)
    x_param.value = np.abs(np.random.randn(n))
    c_param.value = np.abs(np.random.randn(m))

    print("A_param.value:",A_param.value)
    print("x_param.value:",x_param.value)
    print("c_param.value:",c_param.value)

    layer = CvxpyLayer(problem, parameters=[A_param, c_param, x_param], variables=[y], gp=True)
    #==
    torch.random.manual_seed(1)
    A_tch = torch.tensor(init_A)
    A_tch.requires_grad_(True)
    c_tch = torch.tensor(np.squeeze(np.exp(init_c)))
    c_tch.requires_grad_(True)
    train_losses = []
    val_losses = []
    print("A_tch:", np.shape(A_tch))
    print("c_tch:", np.shape(c_tch))

    opt = torch.optim.SGD([A_tch, c_tch], lr=5e-2)
    for epoch in range(1):
        preds = layer(A_tch, c_tch, train_inputs, solver_args={'acceleration_lookback': 0})[0]
        A_tch_constr1 = torch.tanh(1000*(A_tch.sum(axis=1)-1))+1  # guarantee that A_tch * 1^T  <= 1.
        A_tch_constr2 = torch.tanh(1000*(-A_tch))+1 # guarantee that A_tch >= 0.
        c_tch_constr =  torch.tanh(1000*(-c_tch))+1
        print("A_tch:", A_tch)
        print("A_tch_constr2:", A_tch_constr2.sum(axis=1).mean(axis=0))

        loss = (preds - train_outputs).pow(2).sum(axis=1).mean(axis=0)+punish*(A_tch_constr1.mean(axis=0))+punish*(A_tch_constr2.sum(axis=1).mean(axis=0))+punish*(c_tch_constr.mean(axis=0))

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

    #==
    with torch.no_grad():
        train_preds_tch = layer(A_tch, c_tch, train_inputs)[0]
        train_preds = [t.detach().numpy() for t in train_preds_tch]

    #==
    with torch.no_grad():
        val_preds_tch = layer(A_tch, c_tch, val_inputs)[0]
        val_preds = [t.detach().numpy() for t in val_preds_tch]
    return val_preds




def generate_data_pf(num_points, n):
    inputs = np.random.rand(num_points, n)
    # garmma_m = np.array([[0.2,0],[0,0.2]])
    A = np.array([[0.8,0.1],[0.13,0.64]])
    evals_large_sparse, evecs_large_sparse = largest_eigsh(A, 1, which='LM')
    print("pf_eigenvalue:", evals_large_sparse)
    print("evecs_large_sparse:", evecs_large_sparse)

    outputs = []
    for i in range(num_points):
        input_temp = np.diag(inputs[i])*A
        evals_large_sparse, evecs_large_sparse = largest_eigsh(input_temp, 1, which='LM')
        outputs.append(evals_large_sparse)
    inputs_torch = torch.from_numpy(inputs)
    outputs_torch = torch.from_numpy(np.array(outputs))
    return inputs_torch, outputs_torch


generate_data_pf(3, 2)


n = 2
m = 1
# Number of training input-output pairs
N = 2000
# Number of validation pairs
N_val = 100

train_inputs, train_outputs = generate_data_pf(N, n)
val_inputs, val_outputs = generate_data_pf(N_val, n)

lstsq_val_preds,init_A, init_c = monomial_fitting_lstsq(m,n,N,N_val,train_inputs,train_outputs,val_inputs)
# val_preds = monomial_fitting_llcp(m,n**2,train_inputs,train_outputs,val_inputs,init_A.T, init_c)
fig = plt.figure()
val_preds_list = []
lstsq_val_preds_list = []
val_outputs_list = []
i = 0
for j in range(len(val_outputs)):
    # val_preds_list.append(val_preds[j][i])
    lstsq_val_preds_list.append(lstsq_val_preds[j][i])
    val_outputs_list.append(val_outputs[j][i])


print("A:", init_A)
print("c:", init_c)

