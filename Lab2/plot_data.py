import matplotlib.pyplot as plt
import numpy as np

# Initialize error records for each method
error_record_transpose = []
error_record_Pseudoinverse = []
error_record_DLS = []

# Read error records from files data
with open('data/error_record_transpose.txt', 'r') as f:
    for line in f:
        error_record_transpose.append(float(line.strip()))

with open('data/error_record_Pseudoinverse.txt', 'r') as f:
    for line in f:
        error_record_Pseudoinverse.append(float(line.strip()))

with open('data/error_record_DLS.txt', 'r') as f:
    for line in f:
        error_record_DLS.append(float(line.strip()))

# Plotting the error records for each method
dt = 0.01 # Simulation time
time_Pseudoinverse = np.arange(len(error_record_Pseudoinverse)) * dt # Time params for plotting in x axis for Pseudoinverse
time_transpose = np.arange(len(error_record_transpose)) * dt # Time params for plotting in x axis for transpose
time_DLS = np.arange(len(error_record_DLS)) * dt # Time params for plotting in x axis for DLS
plt.figure(figsize=(10, 6))
plt.plot(time_Pseudoinverse, error_record_Pseudoinverse, label="Pseudoinverse", color = 'orange')
plt.plot(time_transpose, error_record_transpose, label="Transpose", color = 'blue')
plt.plot(time_DLS, error_record_DLS, label="DLS", color = 'green')
plt.title("Resolved-rate motion control")
plt.xlabel("Time[s]")
plt.ylabel("Error[m]")
plt.grid()
plt.legend()
plt.show()