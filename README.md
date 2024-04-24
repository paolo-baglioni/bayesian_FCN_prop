# Monte Carlo Bayesian Neural Network Training

This repository implements Monte Carlo methods, particularly Langevin dynamics, for training Bayesian neural networks with a single hidden layer. Currently, the only observable available is the generalization error, but a more comprehensive package will be released soon.

## Usage

To run the code, follow these steps:

1. Clone or download this repository to your local machine.
2. Navigate to the downloaded directory.
3. Execute the following command in your terminal:

```bash
python3 main.py "./data/cifar10_28_500_1000_1000_0.01_0.001_False_1.0_1.0_0.0/parameter.json" > output.txt
```

This command will initiate a Monte Carlo simulation using the CIFAR$10$ dataset with images resized to $28\times28$ pixels. The neural network architecture includes a hidden layer with $N_1=500$, $P=P_{\text{test}}=1000$, temperature $T=0.01$, and a learning rate of $0.001$. The network is trained without biases and without Gaussian noise in the data. Gaussian priors with $\lambda_1=\lambda_0=1.0$ are used. For further information, refer to the parameters specified in the parameter.json file.

For details on parameter notation, please consult our paper [Predictive power of a Bayesian effective action for fully-connected one hidden layer neural networks in the proportional limit](https://arxiv.org/abs/2401.11004)

Additional simulation parameter configurations can be created by utilizing the provided Jupyter notebook script mk_dir.ipynb.

# Bayesian Inference with Normalized Kernel in Proportional Limit

The `proportionalTheory.py` script is a simple Python script that computes Bayesian inference with a normalized kernel in the proportional limit. It operates with the same parameters as described previously, but with the parameter $N_1$ scaling in the range [300,5700,100]. 

## Usage

This script is straightforward to use. Parameters can be directly modified within the script itself. To execute the script, simply run it using a Python interpreter.

```bash
python3 proportionalTheory.py
```

# Contribution
Contributions to this project are welcome! Feel free to open issues for bug reports or feature requests, and submit pull requests for enhancements.

# License
This project is licensed under the MIT License.
