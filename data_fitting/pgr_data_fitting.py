from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_default_tensor_type(torch.DoubleTensor)

##Data generation

n = 2
m = 2
# Number of training input-output pairs
N = 100
# Number of validation pairs
N_val = 50

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
    inputs = 10*np.random.rand(num_points, n)
    F = np.array([[0.8,0.2],[0.3,0.7]])
    # F = transition_matrix(n)
    # F = np.array([[0.7,0.2,0.1],[0.12,0.75,0.23],[0.05,0.15,0.8]])
    outputs = np.dot(F, inputs.T)
    inputs_torch = torch.from_numpy(inputs)
    outputs_torch = torch.from_numpy(outputs.T)
    return inputs_torch, outputs_torch


train_inputs, train_outputs = generate_data(N, 243)
val_inputs, val_outputs = generate_data(N_val, 0)


#==Monomial fit to each component

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


##Fitting
A_param = cp.Parameter(shape=(m, n))
c_param = cp.Parameter(pos=True, shape=(m,))
x_slack = cp.Variable(pos=True, shape=(n,))
x_param = cp.Parameter(pos=True, shape=(n,))
y = cp.Variable(pos=True, shape=(m,))

prediction = cp.multiply(c_param, cp.gmatmul(A_param, x_slack))
objective_fn = cp.sum(prediction / y + y / prediction)

print(objective_fn.log_log_curvature)

constraints = [x_slack >= x_param]
# for i in range(m-1):
#     constraints += [y[i] <= y[i+1]]
problem = cp.Problem(cp.Minimize(objective_fn), constraints)
problem.is_dgp(dpp=True)

A_param.value = np.random.randn(m, n)
x_param.value = np.abs(np.random.randn(n))
c_param.value = np.abs(np.random.randn(m))

layer = CvxpyLayer(problem, parameters=[A_param, c_param, x_param], variables=[y], gp=True)
#==
torch.random.manual_seed(1)
A_tch = torch.tensor(theta.T.value)
A_tch.requires_grad_(True)
c_tch = torch.tensor(np.squeeze(np.exp(log_c.value)))
c_tch.requires_grad_(True)
train_losses = []
val_losses = []
print("A_tch:", np.shape(A_tch))
print("c_tch:", np.shape(c_tch))
lam1 = torch.tensor(1e-1)
lam2 = torch.tensor(1e-1)

opt = torch.optim.SGD([A_tch, c_tch], lr=5e-2)
for epoch in range(1):
    preds = layer(A_tch, c_tch, train_inputs, solver_args={'acceleration_lookback': 0})[0]
    loss = (preds - train_outputs).pow(2).sum(axis=1).mean(axis=0)

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

#==
fig = plt.figure()
val_preds_list = []
lstsq_val_preds_list = []
val_outputs_list = []
i = 0
for j in range(len(val_preds)):
    val_preds_list.append(val_preds[j][i])
    lstsq_val_preds_list.append(lstsq_val_preds[j][i])
    val_outputs_list.append(val_outputs[j][i])

plt.plot(val_preds_list, label='LLCP', color='teal')
plt.plot(lstsq_val_preds_list, label='least squares', linestyle='--', color='red')
plt.plot(val_outputs_list, label='true', linestyle='-.', color='orange')
w, h = 8, 3.5
plt.xlabel(r'$i$')
plt.ylabel(r'$y_i$')
plt.legend()
plt.show()


fig = plt.figure()
val_preds_list = []
lstsq_val_preds_list = []
val_outputs_list = []
i = 1
for j in range(len(val_preds)):
    val_preds_list.append(val_preds[j][i])
    lstsq_val_preds_list.append(lstsq_val_preds[j][i])
    val_outputs_list.append(val_outputs[j][i])

plt.plot(val_preds_list, label='LLCP', color='teal')
plt.plot(lstsq_val_preds_list, label='least squares', linestyle='--', color='red')
plt.plot(val_outputs_list, label='true', linestyle='-.', color='orange')
w, h = 8, 3.5
plt.xlabel(r'$i$')
plt.ylabel(r'$y_i$')
plt.legend()
plt.show()