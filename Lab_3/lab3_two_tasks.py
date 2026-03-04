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
K1 = np.diag([1, 1]) # Control gain matrix
K2 = np.diag([1])

# Desired values of task variables
sigma1_d = np.array([0.0, 1.0]).reshape(2,1) # Position of the end-effector
sigma2_d = np.array([[0.0]]) # Position of joint 1

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
    global sigma1_d
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])
    sigma1_d = np.random.rand(2,1) * 2 - 1 # Random target position in the range [-1, 1]
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
    sigma1 = T[-1][0:2, 3].reshape((2,1))                # Current position of the end-effector
    err1 = sigma1_d - sigma1                  # Error in Cartesian position
    J1 = J[0:2][:]                    # Jacobian of the first task
    DLS_matrix_J1 = DLS(J1, damping_factor)
    P1 = (np.identity(3) - np.linalg.pinv(J1) @ J1)                         # Null space projector
    
    # TASK 2
    sigma2 = q[0]                # Current position of joint 1
    err2 = sigma2_d - sigma2                  # Error in joint position
    J2 = np.array([1, 0, 0]).reshape(1,3)               # Jacobian of the second task
    J2bar = J2 @ P1                  # Augmented Jacobian
    DLS_matrix_J2 = DLS(J2bar, damping_factor)
    
    # Combining tasks
    dq1 = DLS_matrix_J1 @ (K1 @ err1)                    # Velocity for the first task
    dq12 = dq1 + DLS_matrix_J2 @ ((K2 @ err2) - J2 @ dq1)                  # Velocity for both tasks

    q = q.reshape(3,1) + dq12 * dt # Simulation update
    q = q.flatten()

    # Update drawing
    PP = robotPoints2D(T)
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    path.set_data(PPx, PPy)
    point.set_data(sigma1_d[0], sigma1_d[1])

    return line, path, point

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt), 
                                interval=10, blit=True, init_func=init, repeat=True)
plt.show()