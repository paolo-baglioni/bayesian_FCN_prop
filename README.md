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

This command will initiate a Monte Carlo simulation using the CIFAR-10 dataset with images resized to 28x20 pixels. The neural network architecture includes a hidden layer with 500 neurons, $P=P_{\text{test}}=1000$ data points, temperature $T=0.01$, and a learning rate of $0.001$. The network is trained without biases and without Gaussian noise in the data. Gaussian priors with $\lambda_1=\lambda_0=1.0$ are used. For further information, refer to the parameters specified in the parameter.json file.

Additional simulation parameter configurations can be created by utilizing the provided Jupyter notebook script mk_dir.ipynb. For details on parameter notation, please consult the associated paper: XXX.

# Contribution
Contributions to this project are welcome! Feel free to open issues for bug reports or feature requests, and submit pull requests for enhancements.

# License
This project is licensed under the MIT License.
