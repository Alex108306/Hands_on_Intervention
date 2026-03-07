# Import necessary libraries
from lab2_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# Robot definition (3 revolute joint planar manipulator)
d = np.zeros(3)                            # displacement along Z-axis
q = np.array([0.2, 0.5, 0.2])                            # rotation around Z-axis (theta)
a = np.array([0.5, 0.5, 0.5])                   # displacement along X-axis
alpha = np.zeros(3)                        # rotation around X-axis 
revolute = np.array([True, True, True])          # flags specifying the type of joints
K1 = np.diag([1]) # Control gain matrix for the first task
K2 = np.diag([1, 1]) # Control gain matrix for the second task
dq_max = np.array([0.9, 1.0, 0.9]) # Maximum joint velocities

# Record error of ee and joint 1 position
err1_record = []
joint_1_record = []

# Desired values of task variables
sigma1_d = np.array([[0.0]]) # Position of joint 1
sigma2_d = np.array([0.0, 1.0]).reshape(2,1) # Position of the end-effector

# Simulation params
dt = 1.0/60.0
Tt = 10 # Total simulation time
tt = np.arange(0, Tt, dt) # Simulation time vector

# Drawing preparation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2,2))
ax.set_title('Simulation')
ax.set_aspect('equal')
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
ax.grid()
line, = ax.plot([], [], 'o-', lw=2) # Robot structure
path, = ax.plot([], [], 'c-', lw=1) # End-effector path
point, = ax.plot([], [], 'rx') # Target
PPx = []
PPy = []

# Simulation initialization
def init():
    global sigma2_d
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])
    sigma2_d = np.random.rand(2,1) * 2 - 1 # Random target position in the range [-1, 1]
    return line, path, point

# Simulation loop
def simulate(t):
    global q, a, d, alpha, revolute, sigma1_d, sigma2_d
    global PPx, PPy
    
    # Update robot
    T = kinematics(d, q.flatten(), a, alpha)
    J = jacobian(T, revolute)

    # Calculate DLS
    damping_factor = 0.2

    # Update control
    # TASK 1
    sigma1 = q[0]                # Current position of the end-effector
    err1 = sigma1_d - sigma1                  # Error in Cartesian position
    J1 = np.array([1, 0, 0]).reshape(1,3)                   # Jacobian of the first task
    DLS_matrix_J1 = DLS(J1, damping_factor)
    P1 = (np.identity(3) - np.linalg.pinv(J1) @ J1)                         # Null space projector
    
    # TASK 2
    sigma2 = T[-1][0:2, 3].reshape((2,1))                # Current position of joint 1
    err2 = sigma2_d - sigma2                  # Error in joint position
    J2 = J[0:2][:]               # Jacobian of the second task
    J2bar = J2 @ P1                  # Augmented Jacobian
    DLS_matrix_J2 = DLS(J2bar, damping_factor)
    
    # Combining tasks
    dq1 = DLS_matrix_J1 @ (K1 @ err1)                    # Velocity for the first task
    dq12 = dq1 + DLS_matrix_J2 @ ((K2 @ err2) - J2 @ dq1)                  # Velocity for both tasks

    s = np.max(np.abs(dq12) / dq_max) # Scaling factor to ensure joint velocity limits are not exceeded
    if s > 1:
        dq12 = dq12 / s
    else:
        dq12 = dq12

    q = q.reshape(3,1) + dq12 * dt # Simulation update
    q = q.flatten()

    err1_record.append(np.linalg.norm(err2))
    joint_1_record.append(abs(q[0]))

    # Update drawing
    PP = robotPoints2D(T)
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    path.set_data(PPx, PPy)
    point.set_data(sigma2_d[0], sigma2_d[1])

    return line, path, point

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt), 
                                interval=10, blit=True, init_func=init, repeat=True)
plt.show()

# Plot error of end-effector and joint 1 position
t_rec = np.arange(len(err1_record)) * dt # Time vector for recorded joint angles
plt.plot(t_rec, err1_record, label='e1 (end-effector position)')
plt.plot(t_rec, joint_1_record, label='e2 (joint 1 position)')
plt.legend()
plt.title('Task-Priority (two tasks)')
plt.xlabel('Time[s]')
plt.ylabel('Error[1]')
plt.grid()
plt.show()