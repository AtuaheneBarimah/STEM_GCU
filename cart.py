!pip install control

import numpy as np
import matplotlib.pyplot as plt
import control as ctl
from scipy.optimize import minimize

# Define system parameters
m = 0.3  # Mass of the pendulum (kg)
M = 1.0  # Mass of the cart (kg)
L = 1.0  # Length to pendulum center of mass (m)
g = 9.81  # Gravity constant (m/s^2)

# State-space matrices for the linearized inverted pendulum model
A = np.array([[],
              [],
              [],
              []])
B = np.array([[], [], [], [])
C = np.array([[]])
D = np.array([[]])

# Create the state-space system for the inverted pendulum model
pendulum_system = ctl.StateSpace(A, B, C, D)

#Simulation Parameters
# Initial guess for PID parameters
initial_guess = [100, 10, 20]
# Time vector
T = np.linspace(0, 500, 500)
# Input Time
Input_Time = 50
# force applied at t=Input_Time
force_applied = 0.01
# Reference trajectory
ref_trajectory = 0.05


# Define the cost function (Mean Squared Error)
def cost_function(params, T, U):
    Kp, Ki, Kd = params
    PID_A = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    PID_B = np.array([[Kd, Kp, Ki, 0]]).T
    PID_C = np.array([[0, 0, 1, 0]])
    PID_D = np.array([[0]])
    PID_ss = ctl.StateSpace(PID_A, PID_B, PID_C, PID_D)
    system_cl = ctl.series(PID_ss, pendulum_system)
    closed_loop_system = ctl.feedback(system_cl, 1)
    _, yout = ctl.forced_response(closed_loop_system, T, U)
    error = reference_trajectory - yout
    mse = np.mean(error**2)
    mse = mse**(0.5)
    return mse

# Step input at time t=1 second (a disturbance to the pendulum)
t_step = Input_Time
U = np.zeros_like(T)
U[T >= t_step] = force_applied


# Define the reference trajectory
reference_trajectory = np.zeros_like(T)
reference_trajectory[T >= t_step] = ref_trajectory

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
PID_A = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
PID_B = np.array([[Kd_opt, Kp_opt, Ki_opt, 0]]).T
PID_C = np.array([[0, 0, 1, 0]])
PID_D = np.array([[0]])
PID_ss = ctl.StateSpace(PID_A, PID_B, PID_C, PID_D)
system_cl = ctl.series(pendulum_system, PID_ss)
closed_loop_system = ctl.feedback(system_cl, 1)
_, yout = ctl.forced_response(closed_loop_system, T, U)

# Plot the results
plt.figure()
plt.plot(T, yout, label='Pendulum Angle')
plt.plot(T, reference_trajectory, 'r--', label='Reference Trajectory')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.title('Response of Inverted Pendulum with Optimized PID Control')
plt.legend()
plt.grid()
plt.show()
