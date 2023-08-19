#%%
#Load the libraries
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from timeit import default_timer as timer
start=timer()

#Read the data
# D2 is for 16 to 18 and D3 is for 19 to 21, DD contains the dummy for grade progression
DAT1 = pd.read_csv('C:/Users/sabar/OneDrive/Desktop/FORTRAN CODES/PAPER 1/NELS88/nels88/D2.csv')
DAT2 = pd.read_csv('C:/Users/sabar/OneDrive/Desktop/FORTRAN CODES/PAPER 1/NELS88/nels88/D3.csv')
DD = pd.read_csv('C:/Users/sabar/OneDrive/Desktop/FORTRAN CODES/PAPER 1/NELS88/nels88/DD.csv')

#%%
# Likelihood Function
def FUNCT(X, DAT1, DAT2, DD):
    NS = 8609

    # initialize arrays
    L1 = np.zeros((NS, 6, 10))
    L2 = np.ones((NS, 10))
    PR1 = np.zeros((NS, 3, 10))
    C = np.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5])
    CC = X[27] + np.exp(X[28]) * C
    R = np.zeros((NS, 6))
    RXX=np.zeros((NS,6, 10))
    
    Q = np.exp(X[29:38])
    SUMQ = np.sum(Q) + 1.0
    P = np.append(Q / SUMQ, 1 / SUMQ)


    # calculate R)
    R[:, 0] = DAT2 @ X[:14]
    R[:, 1] = DAT1 @ X[:13]
    R[:, 2] = DAT1 @ X[:13]
    R[:, 3] = DAT1 @ X[14:27]
    R[:, 4] = DAT1 @ X[14:27]
    R[:, 5] = DAT1 @ X[14:27]

    # calculate RXX and PR1
    RXX[:,0,:] = R[:, 0, np.newaxis]
    RXX[:,1,:] = R[:, 1, np.newaxis]
    RXX[:,2,:] = R[:, 2, np.newaxis]
    RXX[:,3,:] = R[:, 3, np.newaxis]
    RXX[:,4,:] = R[:, 4, np.newaxis]
    RXX[:,5,:] = R[:, 5, np.newaxis]
    RXX[:,0,:] += C
    RXX[:,1,:] += C
    RXX[:,2,:] += C
    RXX[:,3,:] += CC
    RXX[:,4,:] += CC
    RXX[:,5,:] += CC
    
    PR1 = np.exp(RXX) / (1 + np.exp(RXX))

    # calculate L1
    L1[DD == 1] = PR1[DD == 1]
    L1[DD == 0] = 1 - PR1[DD == 0]

    # calculate L2 and L3
    L2 = L1.prod(axis=1)
    L3 = P @ L2.T

    # calculate log-likelihood
    L = np.log(L3).sum()

    FU = -L/NS
    return FU

#%%
# initialize X
X = np.zeros(38)

# optimization using BFGS method
res = minimize(FUNCT, X, args=(DAT1, DAT2, DD), method = 'BFGS')

# print the result
print(res.x)

NS=8609
# compute THD and THT using vectorized operations
THD = np.sqrt(np.diag(res.hess_inv)) / np.sqrt(NS)
THT = res.x / THD

print('**********************************************************************')
print(THT)

Q = np.exp(res.x[29:38])
SUMQ = np.sum(Q) + 1

P = Q / SUMQ
P = np.append(P, 1/SUMQ)

#%%
#Function to calculate the marginal effects
def marginal(X, P, DAT1, DAT2):
    NS=8609
    NP=38
    
    C = np.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5])
    CC = X[27] + np.exp(X[28]) * C
    
    R = np.zeros((NS, 6))
    R[:, 0] = DAT2 @ X[:14]
    R[:, 1] = DAT1 @ X[:13]
    R[:, 2] = DAT1 @ X[:13]
    R[:, 3] = DAT1 @ X[14:27]
    R[:, 4] = DAT1 @ X[14:27]
    R[:, 5] = DAT1 @ X[14:27]
    
    RXX = np.zeros((NS, 6, 10))
    for k in range(10):
        RXX[:,0,:] = R[:, 0, np.newaxis]
        RXX[:,1,:] = R[:, 1, np.newaxis]
        RXX[:,2,:] = R[:, 2, np.newaxis]
        RXX[:,3,:] = R[:, 3, np.newaxis]
        RXX[:,4,:] = R[:, 4, np.newaxis]
        RXX[:,5,:] = R[:, 5, np.newaxis]
        RXX[:,0,:] += C
        RXX[:,1,:] += C
        RXX[:,2,:] += C
        RXX[:,3,:] += CC
        RXX[:,4,:] += CC
        RXX[:,5,:] += CC

    ME1 = np.zeros((NS, 10, 29))
    for i in range(NS):
        exp_RXX0 = np.exp(RXX[i, 0, :])
        exp_RXX1 = np.exp(RXX[i, 1, :])
        exp_RXX3 = np.exp(RXX[i, 3, :])
        
        ME1[i, :, 13] = (exp_RXX0 / (1 + exp_RXX0) ** 2) * X[13]
        ME1[i, :, :13] = (exp_RXX1[:, np.newaxis] / (1 + exp_RXX1[:, np.newaxis]) ** 2) * X[:13]
        ME1[i, :, 14:27] = (exp_RXX3[:, np.newaxis] / (1 + exp_RXX3[:, np.newaxis]) ** 2) * X[14:27]
        ME1[i, :, 27] = (exp_RXX3 / (1 + exp_RXX3) ** 2) * X[27]
        ME1[i, :, 28] = (exp_RXX3 / (1 + exp_RXX3) ** 2) * X[28] * np.exp(X[28]) * C

    PP=P[:10]
    MU = (P[np.newaxis, :, np.newaxis] * ME1).sum(axis=1)
    MU = MU.mean(axis=0)
    
    MUU = np.zeros(NP)
    for i in range(9):
        MUU[i] = PP[i] * (1 - PP[i]) * X[i+29]
    MUU[:29] = MU
    MUU[29:] = PP[9:] * (1 - PP[9:]) * X[29:]

    return MUU

MUU = marginal(res.x, P, DAT1, DAT2)
print('**********************************************************************')
print(MUU)
#%%
#Draw 100,000 random numbers from multivariate normal with mean MUU and variance Hess-inv
n_samples = 100000
samples = np.random.multivariate_normal(MUU, res.hess_inv, n_samples)

DEV = np.square(MUU - samples)
DEVV = np.sum(DEV, axis=0)
SD = np.sqrt(DEVV / n_samples)
TSTAT = MUU / SD * np.sqrt(n_samples)
print(TSTAT)
end = timer()

print(end - start)