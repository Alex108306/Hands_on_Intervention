from lab6_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.animation as anim
import matplotlib.transforms as trans

PPx_rotate_forward = []
PPy_rotate_forward = []
PPx_forward_rotate = []
PPy_forward_rotate = []
PPx_simultaneously = []
PPy_simultaneously = []
pose_x_rotate_forward = []
pose_y_rotate_forward = []
pose_x_forward_rotate = []
pose_y_forward_rotate = []
pose_x_simultaneously = []
pose_y_simultaneously = []

with open('lab6_mobile_manipulator_simultaneously.txt', 'r') as f:
    next(f) # Skip header
    for line in f:
        time, mobile_x, mobile_y, end_effector_x, end_effector_y = line.strip().split(',')
        PPx_simultaneously.append(float(end_effector_x))
        PPy_simultaneously.append(float(end_effector_y))
        pose_x_simultaneously.append(float(mobile_x))
        pose_y_simultaneously.append(float(mobile_y))

with open('lab6_mobile_manipulator_forward_rotate.txt', 'r') as f:
    next(f) # Skip header
    for line in f:
        time, mobile_x, mobile_y, end_effector_x, end_effector_y = line.strip().split(',')
        PPx_forward_rotate.append(float(end_effector_x))
        PPy_forward_rotate.append(float(end_effector_y))
        pose_x_forward_rotate.append(float(mobile_x))
        pose_y_forward_rotate.append(float(mobile_y))

with open('lab6_mobile_manipulator_rotate_forward.txt', 'r') as f:
    next(f) # Skip header
    for line in f:
        time, mobile_x, mobile_y, end_effector_x, end_effector_y = line.strip().split(',')
        PPx_rotate_forward.append(float(end_effector_x))
        PPy_rotate_forward.append(float(end_effector_y))
        pose_x_rotate_forward.append(float(mobile_x))
        pose_y_rotate_forward.append(float(mobile_y))

# Plotting the end-effector path and distance to the obstacles
plt.figure()
plt.plot(PPx_rotate_forward, PPy_rotate_forward, label='Rotate-then-move', color='orange')
plt.plot(PPx_forward_rotate, PPy_forward_rotate, label='Move-then-rotate', color='blue')
plt.plot(PPx_simultaneously, PPy_simultaneously, label='Simultaneous', color='green')
plt.xlabel('x[m]')
plt.ylabel('y[m]')
plt.title('End-effector path')
plt.legend()
plt.grid()
plt.show()

# Plotting path of the mobile base and end-effector
plt.figure()
plt.plot(pose_x_rotate_forward, pose_y_rotate_forward, label='Rotate-then-move', color='orange')
plt.plot(pose_x_forward_rotate, pose_y_forward_rotate, label='Move-then-rotate', color='blue')
plt.plot(pose_x_simultaneously, pose_y_simultaneously, label='Simultaneous', color='green')
plt.xlabel('x[m]')
plt.ylabel('y[m]')
plt.title('Path of the mobile base')
plt.legend()
plt.grid()
plt.show()