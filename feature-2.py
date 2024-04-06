import numpy as np

# Define the activation functions
def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)

def tanh(x):
    return np.tanh(x)

# Given data
random_values = np.array([-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6])

# Apply the activation functions
relu_values = relu(random_values)
leaky_relu_values = leaky_relu(random_values)
tanh_values = tanh(random_values)

# Print the results
print("ReLU Activation:", relu_values)
print("Leaky ReLU Activation:", leaky_relu_values)
print("Tanh Activation:", tanh_values)

