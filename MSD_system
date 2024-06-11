!pip install control

import numpy as np
import matplotlib.pyplot as plt
import control as ctl
from scipy.optimize import minimize

# Define system parameters
m = 0.5  # Mass
k = 100.0  # Spring constant
b = 0.2  # Damping coefficient

# State-space matrices
A = np.array([[0, 1], [-k/m, -b/m]])
B = np.array([[0], [1/m]])
C = np.array([[1, 0]])
D = np.array([[0]])

# Create the state-space system
system = ctl.StateSpace(A, B, C, D)

# Define the cost function (Mean Squared Error)
def cost_function(params, T, U):
    Kp, Ki, Kd = params
    PID_A = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    PID_B = np.array([[Kd, Kp, Ki]]).T
    PID_C = np.array([[0, 0, 1]])
    PID_D = np.array([[0]])
    PID_ss = ctl.StateSpace(PID_A, PID_B, PID_C, PID_D)
    system_cl = ctl.series(PID_ss, system)
    closed_loop_system = ctl.feedback(system_cl, 1)
    _, yout = ctl.forced_response(closed_loop_system, T, U)
    error = reference_trajectory - yout
    mse = np.mean(error**2)
    return mse

# Initial guess for PID parameters
initial_guess = [350.0, 100.0, 50.0]

# Time vector
T = np.linspace(0, 20, 500)

# Input Time
Input_Time = 5

# Step input at time t=1 second
t_step = Input_Time
U = np.zeros_like(T)
U[T >= t_step] = 5

# Define the reference trajectory
reference_trajectory = np.zeros_like(T)
reference_trajectory[T >= t_step] = Input_Time

# Perform optimization
result = minimize(cost_function, initial_guess, args=(T, U), method='L-BFGS-B')

# Extract optimized PID parameters
Kp_opt, Ki_opt, Kd_opt = result.x

# Print optimized PID parameters
print("Optimized PID parameters:")
print("Kp =", Kp_opt)
print("Ki =", Ki_opt)
print("Kd =", Kd_opt)

# Simulate the response with optimized parameters
PID_A = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
PID_B = np.array([[Kd_opt, Kp_opt, Ki_opt]]).T
PID_C = np.array([[0, 0, 1]])
PID_D = np.array([[0]])
PID_ss = ctl.StateSpace(PID_A, PID_B, PID_C, PID_D)
system_cl = ctl.series(PID_ss, system)
closed_loop_system = ctl.feedback(system_cl, 1)
_, yout = ctl.forced_response(closed_loop_system, T, U)

# Plot the results
plt.figure()
plt.plot(T, yout, label='Output Response')
plt.plot(T, reference_trajectory, 'r--', label='Reference Trajectory')
plt.xlabel('Time (s)')
plt.ylabel('Response')
plt.title('Step Response with Optimized PID Control')
plt.legend()
plt.grid()
plt.show()
