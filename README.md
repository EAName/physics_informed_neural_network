# physics_informed_neural_network
This repository contains an implementation of a Physics-Informed Neural Network (PINN) using TensorFlow to solve a simple second-order ordinary differential equation (ODE):

u′′(x) + u(x)=0, u(0)=0, u(1)=0.

The notebook demonstrates how to incorporate physical laws directly into the training process of a neural network by embedding the differential equation and boundary conditions into the loss function.

###Overview

Physics-informed neural networks leverage both data and the underlying physics to obtain accurate solutions even in regimes where data is sparse. This example uses a PINN to solve the given ODE by:

- Approximating the solution u(x) with a feed-forward neural network.
- Using TensorFlow's automatic differentiation to compute u′′(x).
- Combining data loss and physics-based residual loss to train the network.
- Validating the model through low loss values and visualizations.
