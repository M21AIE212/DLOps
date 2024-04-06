import numpy as np
import matplotlib.pyplot as plt

# Define the activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)

# Generate a range of values
x = np.linspace(-10, 10, 100)

# Plot each activation function in a separate graph
plt.figure(figsize=(12, 9))

# Sigmoid
plt.subplot(2, 2, 1)
plt.plot(x, sigmoid(x), label='Sigmoid', color='blue', linewidth=2)
plt.title('Sigmoid Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)

# Tanh
plt.subplot(2, 2, 2)
plt.plot(x, tanh(x), label='Tanh', color='red', linewidth=2)
plt.title('Tanh Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)

# ReLU
plt.subplot(2, 2, 3)
plt.plot(x, relu(x), label='ReLU', color='green', linewidth=2)
plt.title('ReLU Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)

# Leaky ReLU
plt.subplot(2, 2, 4)
plt.plot(x, leaky_relu(x), label='Leaky ReLU', color='purple', linewidth=2)
plt.title('Leaky ReLU Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)

plt.tight_layout()
plt.show()
