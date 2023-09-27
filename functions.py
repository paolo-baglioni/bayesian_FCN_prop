import json
import torch
import os
import sys
import numpy as np
import torch.nn as nn
from termcolor import colored, cprint
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor
import torchvision.datasets as datasets

def reg_loss( output, target, net, T, lambda0, lambda1 ):
    loss = 0.5*torch.sum(torch.square((output - target))) + ( T * lambda1 ) / 2 * torch.sum(torch.square(net.second_layer.weight)) + ( T * lambda0 ) / 2 * torch.sum(torch.square(net.first_layer.weight))
    return loss

def MNIST_MAKE_DATA(current_working_path, sqrt_N0, P, Ptest):
    
	trans = transforms.Compose([
			transforms.Resize(size=sqrt_N0),
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,)),
			transforms.Lambda(lambda x: torch.flatten(x))
		])
        
	if os.path.exists(current_working_path + "/dataset"):
		cprint("\nCartella dataset gia esistente.", "yellow")
		print("\tCarico il dataset da quello esistente.")
	else:
		cprint("\nCartella dataset non esistente.", "yellow")
		print("\tScarico e carico il dataset.")

	training_data = datasets.MNIST(
    	root =current_working_path + "/dataset",
    	train=True,
    	download=True,
    	transform=trans
	)

	test_data = datasets.MNIST(
    	root = current_working_path + "/dataset",
    	train=False,
    	download=True,
    	transform=trans
	)

	train_dataloader = DataLoader(training_data, batch_size=60000,shuffle=False)
	test_dataloader = DataLoader(test_data, batch_size=10000, shuffle=False)

	x_train, y_train = next(iter(train_dataloader))
	x_test, y_test = next(iter(test_dataloader))

	ones = y_train == torch.ones_like(y_train)
	zeros = y_train == torch.zeros_like(y_train)
	x_train = x_train[ones+zeros][:P]
	y_train = y_train[ones+zeros][:P]

	ones = y_test == torch.ones_like(y_test)
	zeros = y_test == torch.zeros_like(y_test)
	x_test = x_test[ones+zeros][:Ptest]
	y_test = y_test[ones+zeros][:Ptest]

	y_train = y_train[:,None]
	y_test = y_test[:,None]

	print(f"\tsize del dataset di train: \t{x_train.size()} \t- {y_train.size()}")
	print(f"\tsize del dataset di test: \t{x_test.size()} \t- {y_test.size()}")
        
	return x_train, y_train, x_test, y_test



def CFAR_MAKE_DATA(current_working_path, sqrt_N0, P, Ptest):
    
	trans = transforms.Compose([
			transforms.Resize(size=sqrt_N0),
			transforms.ToTensor(),
			transforms.Grayscale(),
			transforms.Normalize((0.1307,), (0.3081,)),
			transforms.Lambda(lambda x: torch.flatten(x))
		])
        
	if os.path.exists(current_working_path + "/dataset"):
		cprint("\nCartella dataset gia esistente.", "yellow")
		print("\tCarico il dataset da quello esistente.")
	else:
		cprint("\nCartella dataset non esistente.", "yellow")
		print("\tScarico e carico il dataset.")

	training_data = datasets.CIFAR10(
    	root = current_working_path + "/dataset",
    	train=True,
    	download=True,
    	transform=trans
	)

	test_data = datasets.CIFAR10(
    	root = current_working_path + "/dataset",
    	train=False,
    	download=True,
    	transform=trans
	)

	train_dataloader = DataLoader(training_data, batch_size=60000,shuffle=False)
	test_dataloader = DataLoader(test_data, batch_size=6000, shuffle=False)

	x_train, y_train = next(iter(train_dataloader))
	x_test, y_test = next(iter(test_dataloader))

	ones = y_train == torch.ones_like(y_train)
	zeros = y_train == torch.zeros_like(y_train)
	x_train = x_train[ones+zeros][:P]
	y_train = y_train[ones+zeros][:P]

	ones = y_test == torch.ones_like(y_test)
	zeros = y_test == torch.zeros_like(y_test)
	x_test = x_test[ones+zeros][:Ptest]
	y_test = y_test[ones+zeros][:Ptest]

	y_train = y_train[:,None]
	y_test = y_test[:,None]

	print(f"\tsize del dataset di train: \t{x_train.size()} \t- {y_train.size()}")
	print(f"\tsize del dataset di test: \t{x_test.size()} \t- {y_test.size()}")
        
	return x_train, y_train, x_test, y_test
