import numpy as np
import matplotlib.pyplot as plt

# Generate a sample 3D point cloud (replace this with your own data)
np.random.seed(0)
sample_size = 100
data_points = np.random.randn(sample_size, 3)  # Assuming 3D data

# Define the characteristic function
def empirical_characteristic_function(t, data):
    return np.mean(np.exp(1j * np.dot(data, t)))

# Range of t values for plotting
t_values = np.linspace(-5, 5, 1000)

# Compute the empirical characteristic function values
ecf_values = [empirical_characteristic_function(t, data_points) for t in t_values]

# Plot the real and imaginary parts of the empirical characteristic function
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(t_values, np.real(ecf_values), label='Real Part')
plt.xlabel('t')
plt.ylabel('Re[ψ(t)]')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t_values, np.imag(ecf_values), label='Imaginary Part')
plt.xlabel('t')
plt.ylabel('Im[ψ(t)]')
plt.legend()

plt.tight_layout()
plt.show()
