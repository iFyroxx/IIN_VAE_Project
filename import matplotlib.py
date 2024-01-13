import matplotlib.pyplot as plt
import numpy as np

# Generate some sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x) + np.cos(x)

# Create a figure with three subplots
plt.figure(figsize=(12, 4))

# Plot the first subplot
plt.subplot(1, 3, 1)
plt.plot(x, y1)
plt.title('Sin(x)')

# Plot the second subplot
plt.subplot(1, 3, 2)
plt.plot(x, y2)
plt.title('Cos(x)')

# Plot the third subplot
plt.subplot(1, 3, 3)
plt.plot(x, y3)
plt.title('Sin(x) + Cos(x)')

# Adjust layout to prevent subplot overlap
plt.tight_layout()

# Show the plot
plt.show()
