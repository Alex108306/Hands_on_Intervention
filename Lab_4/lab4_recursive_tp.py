from lab4_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# Robot model - 3-link manipulator
d = np.zeros(3)                            # displacement along Z-axis
theta = np.array([0.2, 0.5, 0.2])          # rotation around Z-axis
alpha = np.zeros(3)                        # rotation around X-axis
a = np.array([0.5, 0.5, 0.5])              # displacement along X-axis
revolute = np.array([True, True, True])    # flags specifying the type of joints
robot = Manipulator(d, theta, a, alpha, revolute) # Manipulator object
damping_factor = 0.1

# Task hierarchy definition
tasks = [ 
            Position2D("End-effector position", np.array([1.0, 0.5]).reshape(2,1)),
            # Orientation2D("End-effector orientation", np.pi/2),
            # Configuration2D("Configuration", np.array([1.0, 0.5, np.pi/2]).reshape(3,1)),
            JointPosition("Joint position", 0, 0)
        ] 

# Simulation params
dt = 1.0/60.0

# Drawing preparation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2,2))
ax.set_title('Simulation')
ax.set_aspect('equal')
ax.grid()
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
line, = ax.plot([], [], 'o-', lw=2) # Robot structure
path, = ax.plot([], [], 'c-', lw=1) # End-effector path
point, = ax.plot([], [], 'rx') # Target
PPx = []
PPy = []

# Record data for plotting
error_end_effector = []
error_end_effector_orientation = []
error_joint_position = []

# Simulation initialization
def init():
    global tasks
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])
    if len(tasks[0].getDesired()) == 2:
        tasks[0].setDesired((np.random.rand(2,1)*2-1)) # Random target
    else:
        angle = np.array([tasks[0].getDesired()[-1]])
        sigma_d = np.hstack((np.random.rand(1,2)*2-1, angle))
        tasks[0].setDesired(sigma_d.reshape(3,1)) # Random target
    return line, path, point

# Simulation loop
def simulate(t):
    global tasks
    global robot
    global PPx, PPy

    ### Recursive Task-Priority algorithm
    # Initialize null-space 
    P = np.identity(robot.dof)
    # Initialize output vector (joint velocity)
    dq = np.zeros((robot.dof, 1))
    # Loop over tasks
    for i in range(len(tasks)):
        # Update task state
        tasks[i].update(robot)
        # Compute augmented Jacobian
        J_bar = tasks[i].getJacobian() @ P
        # Compute task velocity & Accumulate velocity
        dq = dq + DLS(J_bar, damping_factor) @ (tasks[i].getError() - tasks[i].getJacobian() @ dq)
        # Update null-space projector
        P = P - np.linalg.pinv(J_bar) @ J_bar
    ###

    # Record data for plotting
    if tasks[0].name == "Configuration":
        error_end_effector.append(np.linalg.norm(tasks[0].getError()[0:2]))
        error_end_effector_orientation.append(abs(tasks[0].getError()[2]))
    elif len(tasks) == 1 and tasks[0].name == "End-effector position":
        error_end_effector.append(np.linalg.norm(tasks[0].getError()[0:2]))
    elif len(tasks) > 1 and tasks[1].name == "Joint position":
        error_end_effector.append(np.linalg.norm(tasks[0].getError()[0:2]))
        error_joint_position.append(abs(tasks[1].getError()))
    elif len(tasks) > 1 and tasks[1].name == "End-effector orientation":
        error_end_effector.append(np.linalg.norm(tasks[0].getError()[0:2]))
        error_end_effector_orientation.append(abs(tasks[1].getError())) 


    # Update robot
    robot.update(dq, dt)
    
    # Update drawing
    PP = robot.drawing()
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    path.set_data(PPx, PPy)
    point.set_data(tasks[0].getDesired()[0], tasks[0].getDesired()[1])

    return line, path, point

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt), 
                                interval=10, blit=True, init_func=init, repeat=True)
plt.show()

# Plot errors
tvec = np.arange(0, len(error_end_effector)*dt, dt)
plt.figure()
if tasks[0].name == "Configuration":
    plt.plot(tvec, error_end_effector, label='End-effector position error')
    plt.plot(tvec, error_end_effector_orientation, label='End-effector orientation error')
elif len(tasks) == 1 and tasks[0].name == "End-effector position":
    plt.plot(tvec, error_end_effector, label='End-effector position error')
elif len(tasks) > 1 and tasks[1].name == "Joint position":
    plt.plot(tvec, error_end_effector, label='End-effector position error')
    plt.plot(tvec, error_joint_position, label='Joint position error')
elif len(tasks) > 1 and tasks[1].name == "End-effector orientation":
    plt.plot(tvec, error_end_effector, label='End-effector position error')
    plt.plot(tvec, error_end_effector_orientation, label='End-effector orientation error')
plt.xlabel('Time step')
plt.ylabel('Error')
plt.title('Task Errors')
plt.legend()
plt.grid()
plt.show()