import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    'text.usetex': True,  # Enable LaTeX rendering
    'font.size': 14,      # Set global font size
    'font.family': 'serif'
})

# Constants
r_0 = 1.04e-3  # 1.04 mm in meters
f = 10e6       # 10 MHz
omega = 2 * np.pi * f
e = 1.60218e-19  # Elementary charge in C
m = 87.9056 * 1.66054e-27  # Mass of 88Sr+ in kg

# Load the data
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "Mathieu scan data BICEPS")
raw_data = os.path.join(data_dir, "121025_BICEPS_mathieu_scan.csv")
data = pd.read_csv(raw_data, skiprows=0, delimiter=',', header=0)

# Use the correct column names
rf_voltage = data['RF_Voltage(V)']
dc_voltage = data['DC_Voltage(V)']
stability = data['Stability_Percentage']

# Calculate Mathieu parameters
a = (8 * e * dc_voltage) / (m * r_0**2 * omega**2)
q = (2 * e * rf_voltage) / (m * r_0**2 * omega**2)

# --- Contour Plot in Mathieu Parameters ---
a_unique = np.unique(a)
q_unique = np.unique(q)
a_grid, q_grid = np.meshgrid(a_unique, q_unique)
stability_grid = np.zeros_like(a_grid, dtype=float)

for i, a_val in enumerate(a_unique):
    for j, q_val in enumerate(q_unique):
        mask = (np.isclose(a, a_val)) & (np.isclose(q, q_val))
        if np.any(mask):
            stability_grid[j, i] = stability[mask].values[0]

plt.figure(figsize=(8, 6))

# Contour plot in Mathieu parameters
contour = plt.contourf(q_grid, a_grid, stability_grid, levels=20, cmap='viridis')
plt.colorbar(contour, label='Stability Percentage')
plt.xlabel('Mathieu Parameter $q$')
plt.ylabel('Mathieu Parameter $a$')
plt.title('Simulated Mathieu Stability Diagram')

# Add theoretical points
plt.scatter(0.908, 0, color='red', marker='*', s=50, label='Theory: $q_{max} = (0.908, 0)$')
plt.scatter(0.706, 0.237, color='blue', marker='*', s=50, label='Theory: $a_{max} = (0.706, 0.237)$')

plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'Mathieu Stability Diagram Contour Plot.png'), format='PNG', dpi=200)
plt.show()

# --- Scatter Plot in Mathieu Parameters ---
# plt.subplot(1, 2, 2)
# scatter = plt.scatter(a, q, c=stability, cmap='viridis', s=5)
# plt.colorbar(scatter, label='Stability Percentage (%)')
# plt.xlabel('Mathieu Parameter $a$')
# plt.ylabel('Mathieu Parameter $q$')
# plt.title('Mathieu Stability Diagram (Scatter in Mathieu Parameters)')

# plt.tight_layout()
# plt.savefig(os.path.join(data_dir, 'Mathieu Stability Diagram Mathieu Parameters.png'), format='PNG', dpi=200)
# plt.show()

