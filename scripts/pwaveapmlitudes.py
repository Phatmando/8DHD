import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# Define parameters
n_points = 100
theta = np.linspace(0, 2 * np.pi, n_points)
phi = np.linspace(0, 2 * np.pi, n_points)
theta, phi = np.meshgrid(theta, phi)
r = 1 / np.sqrt(2)  # Radius for 4D Clifford torus embedded in R^8

# Parameterize the 4D torus (a slice of the 8D torus)
x1 = r * np.cos(theta)
x2 = r * np.sin(theta)
x3 = r * np.cos(phi)
x4 = r * np.sin(phi)

# Define a prime-based scalar field for coloring
primes = [2, 3, 5, 7]
f = np.cos(primes[0] * theta) + np.cos(primes[1] * phi)  # Prime wave function
f_min, f_max = np.min(f), np.max(f)
f_norm = (f - f_min) / (f_max - f_min)  # Normalize for colormap
colors = cm.plasma(f_norm)  # Use 'plasma' for vibrant contrast

# Stereographic projection from R^4 to R^3
def stereographic_projection(x1, x2, x3, x4):
    denom = 1 - x4
    X = x1 / denom
    Y = x2 / denom
    Z = x3 / denom
    return X, Y, Z

# Project coordinates to 3D
X, Y, Z = stereographic_projection(x1, x2, x3, x4)

# Create figure and 3D axes
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the torus surface with enhanced visuals
surface = ax.plot_surface(X, Y, Z, facecolors=colors, alpha=0.6, rstride=1, cstride=1,
                         shade=True, lightsource=mcolors.LightSource(azdeg=30, altdeg=65))

# Plot prime strings (curves on the surface)
t = np.linspace(0, 2 * np.pi, 1000)
string_configs = [(2, 3, 'red'), (3, 5, 'yellow')]  # Prime pairs and colors
for p, q, color in string_configs:
    theta_string = p * t
    phi_string = q * t
    x1_string = r * np.cos(theta_string)
    x2_string = r * np.sin(theta_string)
    x3_string = r * np.cos(phi_string)
    x4_string = r * np.sin(phi_string)
    X_string, Y_string, Z_string = stereographic_projection(x1_string, x2_string, x3_string, x4_string)
    ax.plot(X_string, Y_string, Z_string, color=color, linewidth=2.5, label=f'String ({p},{q})')

# Add colorbar for scalar field interpretation
norm = plt.Normalize(f_min, f_max)
sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=10, pad=0.1)
cbar.set_label('Prime Wave Amplitude', fontsize=12)

# Customize plot for clarity and aesthetics
ax.set_xlabel('X', fontsize=10)
ax.set_ylabel('Y', fontsize=10)
ax.set_zlabel('Z', fontsize=10)
ax.set_title('3D Projection of 4D Torus Slice with Prime Waves', fontsize=14, pad=20)
ax.legend(loc='upper right', fontsize=10)
ax.view_init(elev=25, azim=40)  # Adjusted for optimal viewing
ax.grid(False)  # Remove grid for cleaner look

# Adjust layout and display
plt.tight_layout()
plt.show()