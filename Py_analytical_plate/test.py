import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Example data points
x = np.array([1, 2, 4, 5, 8])
y = np.array([1, 4, 9, 16, 25])

# Quadratic spline interpolation
quadratic_interpolation = interp1d(x, y, kind='quadratic')

# Generate new x values for a smooth curve
x_new = np.linspace(x.min(), x.max(), 500)
y_new = quadratic_interpolation(x_new)

# Plot the original data points
plt.scatter(x, y, color='red', label='Original Points')

# Plot the interpolated smooth curve
plt.plot(x_new, y_new, color='blue', label='Quadratic Spline Interpolation')

# Labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Parabolic Arc through Interpolated Points')
plt.legend()
plt.grid(True)
plt.show()
