import numpy as np
import matplotlib.pyplot as plt

# Generate a Laplace distribution with mean=0, scale=1
data = np.random.laplace(loc=0, scale=1, size=1000)

# Plotting the distribution
plt.hist(data, bins=30, density=True, alpha=0.6, color='b')
plt.title("Laplace Distribution (High Kurtosis)")
plt.show()


# Generate a Student's t-distribution with df=2
data = np.random.standard_t(df=2, size=1000)

# Plotting the distribution
plt.hist(data, bins=30, density=True, alpha=0.6, color='g')
plt.title("Student's t-Distribution (High Kurtosis)")
plt.show()


data = np.random.normal(size=1000)

# Plotting the distribution
plt.hist(data, bins=30, density=True, alpha=0.6, color='g')
plt.title("Normal distribution")
plt.show()