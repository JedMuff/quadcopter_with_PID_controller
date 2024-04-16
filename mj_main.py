
# Import libraries
import numpy as np
import random
import math

# Import custom quadcopter and controller classes
from MjQuadcopter import Quadcopter
from PID_Controller import PID_Controller

#sim run time
sim_start = 0 #start time of simulation
sim_end = 15 #end time of simulation in sec
dt = 0.01 #step size in sec
time_index = np.arange(sim_start, sim_end + dt, dt)

# Initial conditions
r_ref = np.array([0., 0.5, 3.]) # desired position [x, y, z] in inertial frame - meters

#initial conditions
#pos = np.array([0., 0., 3.]) # starting location [x, y, z] in inertial frame - meters
#pos = [0.5, 1., 2.] # starting location [x, y, z] in inertial frame - meters
pos = [0., 0., 0.] # starting location [x, y, z] in inertial frame - meters
vel = np.array([0., 0., 0.]) #initial velocity [x; y; z] in inertial frame - m/s
ang = np.array([0., 0., 0.]) #initial Euler angles [phi, theta, psi] relative to inertial frame in deg

# Add initial random roll, pitch, and yaw rates
# deviation = 10 # magnitude of initial perturbation in deg/s
# random_set = np.array([random.random(), random.random(), random.random()])
# ang_vel = np.deg2rad(2* deviation * random_set - deviation) #initial angular velocity [phi_dot, theta_dot, psi_dot]
ang_vel = np.array([0.0, 0.0, 0.0]) #initial angular velocity [phi_dot, theta_dot, psi_dot]

gravity = 9.8 # acceleration due to gravity, m/s^2

# Gains for position controller
Kp_pos = [.95, .95, 15.] # proportional [x,y,z]
Kd_pos = [1.8, 1.8, 15.]  # derivative [x,y,z]
Ki_pos = [0.2, 0.2, 1.0] # integral [x,y,z]
Ki_sat_pos = 1.1*np.ones(3)  # saturation for integral controller (prevent windup) [x,y,z]

# Gains for angle controller
Kp_ang = [6.9, 6.9, 25.] # proportional [x,y,z]
Kd_ang = [3.7, 3.7, 9.]  # derivative [x,y,z]
Ki_ang = [0.1, 0.1, 0.1] # integral [x,y,z]
Ki_sat_ang = 0.1*np.ones(3)  # saturation for integral controller (prevent windup) [x,y,z]

# Create quadcotper with position and angle controller objects
quadcopter = Quadcopter(pos,vel,ang,ang_vel,r_ref)
pos_controller = PID_Controller(Kp_pos, Kd_pos, Ki_pos, Ki_sat_pos, dt)
angle_controller = PID_Controller(Kp_ang, Kd_ang, Ki_ang, Ki_sat_ang, dt)

max_angle = math.pi/12 #radians, max angle allowed at any time step

# Simulation
for time in enumerate(time_index):

    #find position and velocity error and call positional controller
    pos_error = quadcopter.calc_pos_error()
    vel_error = quadcopter.calc_vel_error()
    des_acc = pos_controller.control_update(pos_error,vel_error)
    
    #Modify z gain to include thrust required to hover
    des_acc[2] = (gravity + des_acc[2])/(math.cos(quadcopter.angle[0]) * math.cos(quadcopter.angle[1]))
    
    #calculate thrust needed  
    thrust_needed = quadcopter.mass * des_acc[2]

    #Check if needed acceleration is not zero. if zero, set to one to prevent divide by zero below
    mag_acc = np.linalg.norm(des_acc)
    if mag_acc == 0:
        mag_acc = 1
    
    #use desired acceleration to find desired angles since the quad can only move via changing angles
    print(f"-des_acc[1]: {-des_acc[1]}")
    print(f"mag_acc: {mag_acc}")
    print(f"quadcopter.angle[1]: {quadcopter.angle[1]}")
    print(-des_acc[1] / mag_acc / math.cos(quadcopter.angle[1]))
    ang_des = [math.asin(-des_acc[1] / mag_acc / math.cos(quadcopter.angle[1])),
        math.asin(des_acc[0] / mag_acc),
         0]

    #check if exceeds max angle
    mag_angle_des = np.linalg.norm(ang_des)
    if mag_angle_des > max_angle:
        ang_des = (ang_des / mag_angle_des) * max_angle

    #call angle controller
    quadcopter.angle_ref = ang_des
    ang_error = quadcopter.calc_ang_error()
    ang_vel_error = quadcopter.calc_ang_vel_error()
    tau_needed = angle_controller.control_update(ang_error, ang_vel_error)

    #Find motor speeds needed to achieve desired linear and angular accelerations
    quadcopter.des2speeds(thrust_needed, tau_needed)

    # Step in time and update quadcopter attributes
    quadcopter.step()