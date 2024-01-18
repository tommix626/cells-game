import matplotlib.pyplot as plt
import numpy as np

# Create a scatter plot with 200 points
x = np.random.randn(200)
y = np.random.randn(200)

# Create a colormap with 20 colors
cmap = plt.cm.tab20

# Color the points according to the colormap
colors = cmap(np.arange(200) % 20)

# Plot the scatter plot
plt.scatter(x, y, c=colors)
plt.colorbar()
plt.show()
