import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
  """Sigmoid activation function."""
  return 1 / (1 + np.exp(-x))

def relu(x):
  """ReLU (Rectified Linear Unit) activation function."""
  return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
  """Leaky ReLU activation function with a slope of alpha for negative values."""
  return np.maximum(alpha * x, x)

def tanh(x):
  """Tanh (Hyperbolic Tangent) activation function."""
  return np.tanh(x)

# Create a range of input values
x = np.linspace(-5, 5, 100)  # Adjust the range as needed

# Generate activation function outputs
y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)
y_tanh = tanh(x)

# Create the plot
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

# Plot each activation function with a different label
plt.plot(x, y_sigmoid, label='Sigmoid')
plt.plot(x, y_relu, label='ReLU')
plt.plot(x, y_leaky_relu, label='Leaky ReLU')
plt.plot(x, y_tanh, label='Tanh')

# Add labels and title
plt.xlabel('Input (x)')
plt.ylabel('Output (y)')
plt.title('Activation Functions')

# Add legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
