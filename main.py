import json
import torch
import os
import sys
import time
import numpy as np
import torch.nn as nn
from npy_append_array import NpyAppendArray
from termcolor import colored, cprint
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor
import torchvision.datasets as datasets
import classes
import functions

start_time_overhead = time.time()

if torch.cuda.is_available():
  device = torch.device("cuda:0")
  print("Running on the GPU")
else:
  device = torch.device("cpu")
  print("Running on the CPU")

print(f"DEVICE : {device}")

parameter_file = sys.argv[1]
if os.path.exists(parameter_file):
    print("\n")
    cprint("File di parametri trovato. ", "red")
    print("\tCarico i parametri da : " + parameter_file)
    working_path = parameter_file[0:-14]
else:
    print("\n")
    cprint("File di parametri non trovato.", "red")
    sys.exit("Esecuzione abortita.")

with open(parameter_file) as f:
    parameter = json.load(f)

count = 1

string_output_file_train = working_path + "train_loss_" + str(count) + ".npy" 
string_output_file_test  = working_path + "test_loss_" + str(count) + ".npy"

flag_for_output = os.path.exists(string_output_file_train)
while flag_for_output:
    count = count + 1
    string_output_file_train = working_path + "train_loss_" + str(count) + ".npy" 
    string_output_file_test  = working_path + "test_loss_" + str(count) + ".npy"
    flag_for_output = os.path.exists(string_output_file_train)
    

seed = abs(int(hash(working_path)/10000000000)) + count
    
cprint("\nParametri caricati:", "green")
for (param_name, param_value) in parameter.items():
    print("\t" + str(param_name) + "\t" + " : " + str(param_value))

print("\t" + "seed:" + "\t" + " : " + str(seed))

sqrt_N0		=	parameter["sqrt_N0"]
N0 			=	parameter["N0"]	
N1 			=	parameter["N1"]			
P			=	parameter["P"]
Ptest		=	parameter["Ptest"]
T			=	parameter["T"]
epsilon		=	parameter["epsilon"]
bias 		= 	parameter["bias"]
NUM_MIS		=	parameter["NUM_MIS"]
sv_step		=	parameter["sv_step"]
BUFFER_SIZE	=	parameter["BUFF_S"]
lambda0     =   parameter["lambda0"]
lambda1     =   parameter["lambda1"]
dataset     =   parameter["dataset"]
noise       =   parameter["noise"]

coeff = np.sqrt(2 * epsilon * T)

number_of_emptyings = int( NUM_MIS/BUFFER_SIZE )
epochs = number_of_emptyings * BUFFER_SIZE * sv_step
cprint(f"\tepochs\t : {epochs}" , "magenta")

if dataset == "mnist":
    (x_train, y_train, x_test, y_test) = functions.MNIST_MAKE_DATA("./", sqrt_N0, P, Ptest, noise)
elif dataset== "cifar10":
    (x_train, y_train, x_test, y_test) = functions.CFAR_MAKE_DATA("./", sqrt_N0, P, Ptest, noise)
else:
    cprint("Errore: dataset non conosciuto.", "magenta")
    exit()

x_train = x_train.to(device)
y_train = y_train.to(device)
x_test = x_test.to(device)
y_test = y_test.to(device)

torch.manual_seed(seed)

model = classes.NeuralNetwork(N0, N1, bias)
model = model.to(device)
optimizer = classes.LangevinMC(model.parameters(),epsilon=epsilon, temperature=T)

cprint("\nStai utilizzando una NN cosi fatta:", "blue")
print(f"{model}")

criterion = functions.reg_loss
bare_criterion = nn.MSELoss(reduction='mean')

loss_history_train	=	torch.zeros(BUFFER_SIZE).to(device)
loss_history_test	=	torch.zeros(BUFFER_SIZE).to(device)

output_file_train = NpyAppendArray(string_output_file_train)
output_file_test  = NpyAppendArray(string_output_file_test)

end_time_overhead = time.time()

model.train()
print(f"\nPartita la simulazione di Monte-Carlo ..." )

start_time_mc = time.time()


y_pred = torch.empty((P,1)).to(device)

for i in range(number_of_emptyings):
    for j in range(BUFFER_SIZE):
        for k in range(sv_step):
            optimizer.zero_grad()
            y_pred = model.forward(x_train)
            loss = criterion(y_pred, y_train, model,T, lambda0, lambda1)
            loss.backward()
            optimizer.step()
		
        loss_history_train[j] = loss.item()
        loss_history_test[j] = bare_criterion(model.forward(x_test), y_test).item()
    
    buffer_1 = loss_history_train.cpu().numpy()
    buffer_2 = loss_history_test.cpu().numpy()
    output_file_train.append(	buffer_1	)
    output_file_test.append(	buffer_2	)
                
    
end_time_mc = time.time()
print(f"\nRun-time (overhead)\t: {end_time_overhead-start_time_overhead}\tsec." )
print(f"Run-time (monte-carlo)\t: {end_time_mc-start_time_mc}\tsec. \n" )