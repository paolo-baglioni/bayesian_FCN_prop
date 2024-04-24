import math
import numpy as np
from scipy.optimize import minimize
import functions as fn
from matplotlib import pyplot as plt
from termcolor import colored, cprint
import torch

def S(x):
    HH = T * np.identity(P) + (x/lambda1) * diag_K
    HH_inv = np.linalg.inv(HH)
    part_1 = -1 + x + np.log(1/x)
    part_2 = alpha1/P * np.sum(np.log(T + x*K_eigval/lambda1))
    part_3 = alpha1 * lambda1 / P * np.dot(ytilde, np.dot(HH_inv, ytilde))
    return (part_1 + part_2 + part_3)

sqrt_N0  = 28
N0       = int( 28*28 )
P		 = 1000
Ptest    = 1000
lambda0  = 1.0
lambda1  = 1.0
noise	 = 0.0
T		 = 0.01
dataset  = "cifar10"

cprint("\nParameters:"					, color="red")
cprint(f"\t sqrt N0 : \t {sqrt_N0}"		, color="red")
cprint(f"\t N0      : \t {N0}"			, color="red")
cprint(f"\t P       : \t {P}"			, color="red")
cprint(f"\t Ptest   : \t {Ptest}"		, color="red")
cprint(f"\t lambda0 : \t {lambda0}"		, color="red")
cprint(f"\t lambda1 : \t {lambda1}"		, color="red")
cprint(f"\t noise   : \t {noise}"		, color="red")
cprint(f"\t T       : \t {T}"			, color="red")
cprint(f"\t dataset : \t {dataset}"		, color="red")

N1_list = np.arange(300,5700,100)

num_of_N1 = len(N1_list)
alpha     = P / N1_list[0]
cprint("\nN1_list:", "green")
print(N1_list)

epsilon_g = np.zeros( num_of_N1 )

(x_train, y_train, x_test, y_test) = fn.CIFAR_MAKE_DATA("./dataset_cifar10/", sqrt_N0, P, Ptest, noise)
x = np.squeeze(x_train.numpy())
y = np.squeeze(y_train.numpy())
xt  = np.squeeze(x_test.numpy())
yt  = np.squeeze(y_test.numpy())

C = np.zeros((P,P))
for i in range(P):
	for j in range(P):
		C[i,j] = np.dot(x[i],x[j])/(lambda0 * N0)

K = np.zeros((P,P))
for i in range(P):
	for j in range(P):
		K[i,j] = (2/np.pi)*np.arcsin(2 * C[i,j] / np.sqrt((1+2*C[i,i])*(1+2*C[j,j])) )

K_inv = np.linalg.inv(K)

[K_eigval, K_eigvec] = np.linalg.eig(K)
U = K_eigvec
Udag = np.transpose(U)
diag_K = np.diagflat(K_eigval)
diag_K_inv = np.linalg.inv(diag_K)
ytilde = np.dot( Udag, y)

for k in range(num_of_N1):

	N1 = N1_list[k]
	alpha1  = P/N1

	x0 = 1.0
	bns = ((1e-8,np.inf),)

	res = minimize(S, x0, bounds=bns, tol=1e-20)
	Q_bar = res.x
	Q = 1/Q_bar -1
	
	epsilon_g_temp = np.zeros(Ptest)

	for j in range(Ptest):
		
		C0 = np.zeros(P)
		for i in range(P):
			C0[i] = np.dot(xt[j],x[i])/(lambda0 * N0)
		C00 = np.dot(xt[j],xt[j])/(lambda0 * N0)
		K0 = np.zeros(P)
		
		for ii in range(P):
			K0[ii] = (2/np.pi) * np.arcsin(2 * C0[ii] / np.sqrt((1+2*C[ii,ii])*(1+2*C00)) )
		K00 = (2/np.pi) * np.arcsin(2 * C00 / np.sqrt((1+2*C00)*(1+2*C00)) )
	
		SM = (T * np.identity(P) + (Q_bar/lambda1)*K)
		SM_inv = np.linalg.inv(SM)
		
		GAMMA1 = (Q_bar/lambda1) * np.dot(K0, np.dot(SM_inv, y))
		SIGMA1SQ = (Q_bar/lambda1) * (K00 - (Q_bar/lambda1) * np.dot(K0, np.dot(SM_inv, K0)))
		
		epsilon_g_temp[j] = (yt[j] - GAMMA1)**2 + SIGMA1SQ

	epsilon_g[k] = np.mean(epsilon_g_temp)

	cprint(f"{k}.\tIteration (N1) : ({N1}) --> [Q, Qbar] = [{float(Q)},{float(Q_bar)}] --> epsilon_g : {epsilon_g[k]}", "blue")

np.save("./result.npy", epsilon_g)