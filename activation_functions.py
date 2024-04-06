import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def relu(x):
  return x * (x > 0)

def leaky_relu(x, alpha=0.01):
  return x * (x > 0) + alpha * x * (x <= 0)

def tanh(x):
  return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

# Generate random data
random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]

# Generate plots for each activation function
x = np.linspace(-5, 5, 100)  # Create a range of x-values for plotting

plt.figure(figsize=(10, 6))

plt.plot(x, sigmoid(x), label='Sigmoid')
plt.plot(x, relu(x), label='ReLU')
plt.plot(x, leaky_relu(x), label='Leaky ReLU')
plt.plot(x, tanh(x), label='Tanh')

plt.xlabel('Input Value')
plt.ylabel('Output Value')
plt.title('Activation Functions')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

