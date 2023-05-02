import torch
import gpytorch

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a Gaussian Process Model

class DeepKernelRegression(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(DeepKernelRegression, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)),
                num_dims=2, grid_size=100
            )
            # self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=2)

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
        
        def fit(self, data, args):
            train_x, train_y = data[0], data[1]
             # Set the model in training mode
            self.train()
            self.likelihood.train()

            # Train the model using Adam optimizer
            optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

            training_iter = 500
            for i in range(training_iter):
                optimizer.zero_grad()
                output = self.forward(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()
            
            fit_loss = gpytorch.metrics.mean_squared_error(output, train_y, squared=True).item()
            return loss.item(), fit_loss, None

        def predict(self, dataloader):
        
            # Set model and likelihood in evaluation mode
            self.eval()
            self.likelihood.eval()

            # Make predictions
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                if isinstance(dataloader, tuple):
                    x, _ = dataloader
                else:
                    x = dataloader

                output = self(x)
                predictions = self.likelihood(output)

            return {
                'predictions': predictions,
                'mean': predictions.mean,
                'stddev': predictions.stddev.detach(),
            }