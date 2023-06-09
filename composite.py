import math

import torch
import numpy as np
import gpytorch
from matplotlib import pyplot as plt
import seaborn as sns

from simulators.base import BaseSimulator
from simulators.analytical_simulators import StepFunction, Motorcycle

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a Gaussian Process Model

class DeepKernelRegression(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(DeepKernelRegression, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            # self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            #     gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)),
            #     num_dims=2, grid_size=100
            # )
            self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=2)

            class LargeFeatureExtractor(torch.nn.Sequential):
                def __init__(self, data_dim):
                    super(LargeFeatureExtractor, self).__init__()
                    self.add_module('linear1', torch.nn.Linear(data_dim, 1000))
                    self.add_module('relu1', torch.nn.ReLU())
                    self.add_module('linear2', torch.nn.Linear(1000, 500))
                    self.add_module('relu2', torch.nn.ReLU())
                    self.add_module('linear3', torch.nn.Linear(500, 50))
                    self.add_module('relu3', torch.nn.ReLU())
                    self.add_module('linear4', torch.nn.Linear(50, 2))

            self.feature_extractor = LargeFeatureExtractor(data_dim=train_x.size(-1))

            # This module will scale the NN features so that they're nice values
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

        def forward(self, x):
            # We're first putting our data through a deep net (feature extractor)
            projected_x = self.feature_extractor(x)
            projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"

            mean_x = self.mean_module(projected_x)
            covar_x = self.covar_module(projected_x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class CompositeGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(CompositeGPModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        
         # Matérn kernel with smoothness parameter ν = 1/2
        self.rbf_kernel1 = gpytorch.kernels.RBFKernel()
        self.rbf_kernel2 = gpytorch.kernels.RBFKernel()
        self.covar_kernel = gpytorch.kernels.AdditiveKernel(self.rbf_kernel1, self.rbf_kernel2)

    def forward(self, x):
        mean_x = self.mean_module(x)

        # Composite kernel: sum of the two RBF kernels
        covar_x = self.covar_kernel(x)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Training data
    sim = Motorcycle()
    SEED = 1
    x, y = sim.sample_initial_data(50, 'random', SEED)
    train_x = x.to(device)
    train_y = y.to(device)
    
    # Initialize the likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = DeepKernelRegression(train_x, train_y, likelihood).to(device)

    # Set the model in training mode
    model.train()
    likelihood.train()

    # Train the model using Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iter = 1000
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        print(model.covar_module.lengthscale)
        print(likelihood.noise.item())

        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))

    # Set the model in evaluation mode
    model.eval()
    likelihood.eval()

    # Test points
    x = sim.search_space()
    y = sim.mean(x)
    std = sim.stddev(x)
    test_x = torch.tensor(x).reshape(-1).to(device).float()
    test_y = torch.tensor(y).reshape(-1).to(device).float()
    test_var = torch.tensor(std).reshape(-1).to(device).float()

    # Predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))

    # Plot
    
    plt.figure()
    
    # original simulator
    sns.lineplot(x=test_x.cpu().numpy(), y=test_y.cpu().numpy(), color='blue', label="Ground truth")
    plt.fill_between(test_x.cpu().numpy(), test_y.cpu().numpy() - test_var.cpu().numpy(), test_y.cpu().numpy() + test_var.cpu().numpy(), color='blue', alpha=0.2)

    plt.plot(test_x.cpu().numpy(), observed_pred.mean.cpu().numpy(), color='yellow', label='Predicted Value')
    plt.fill_between(test_x.cpu().numpy(), 
                    observed_pred.confidence_region()[0].cpu().numpy(),
                    observed_pred.confidence_region()[1].cpu().numpy(), 
                    color = 'orange', alpha=0.5)
    # plt.plot(test_x.cpu().numpy(), test_y.cpu().numpy(), 'r--')
    plt.title('Deep Kernel Regression')
    plt.legend(loc='upper left')
    
    plt.savefig('composite.png')