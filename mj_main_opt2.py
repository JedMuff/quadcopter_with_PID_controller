
# Import libraries
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import scipy.optimize
import opal_py3 as opal

# Import custom quadcopter and controller classes
from MjQuadcopter_opt2 import Quadcopter
from PID_Controller import PID_Controller

#sim run time
sim_start = 0 #start time of simulation
sim_end = 12 #end time of simulation in sec
dt = 0.01 #step size in sec
time_index = np.arange(sim_start, sim_end + dt, dt)

# Initial conditions
# r_ref = np.random.uniform(0.1,3,3)# desired position [x, y, z] in inertial frame - meters
r_ref = np.array([1., 0.5, 0.5]) # desired position [x, y, z] in inertial frame - meters

#initial conditions
#pos = np.array([0., 0., 3.]) # starting location [x, y, z] in inertial frame - meters
#pos = [0.5, 1., 2.] # starting location [x, y, z] in inertial frame - meters
pos = [0.0, 0.0, 0.0] # starting location [x, y, z] in inertial frame - meters
vel = np.array([0., 0., 0.]) #initial velocity [x; y; z] in inertial frame - m/s
ang = np.array([0., 0., 0.]) #initial Euler angles [phi, theta, psi] relative to inertial frame in deg

# Add initial random roll, pitch, and yaw rates
# deviation = 10 # magnitude of initial perturbation in deg/s
# random_set = np.array([random.random(), random.random(), random.random()])
# ang_vel = np.deg2rad(2* deviation * random_set - deviation) #initial angular velocity [phi_dot, theta_dot, psi_dot]
ang_vel = np.array([0.0, 0.0, 0.0]) #initial angular velocity [phi_dot, theta_dot, psi_dot]

gravity = 9.81 # acceleration due to gravity, m/s^2
def obj_fun(params, target=1.0,render=False):
    # Gains for position controller
    #Kp_pos = [100.95, 100.95, 656.6] # proportional [x,y,z]
    #Kd_pos = [20.8, 20.8, 15.]  # derivative [x,y,z]
    #Ki_pos = [1.2, 1.2, 1.0] # integral [x,y,z]
    Kp_pos = [params[0], params[1], params[2]] # proportional [x,y,z]
    Kd_pos = [params[3], params[4], params[5]]  # derivative [x,y,z]
    Ki_pos = [params[6], params[7], params[8]] # integral [x,y,z]

    Ki_sat_pos = 0.1*np.ones(3)  # saturation for integral controller (prevent windup) [x,y,z]

    # Gains for angle controller
    #Kp_ang = [56.9, 56.9, 25.] # proportional [x,y,z]
    #Kd_ang = [7.7, 7.7, 9.]  # derivative [x,y,z]
    #Ki_ang = [0.5, 0.5, 0.1] # integral [x,y,z]
    Kp_ang = [params[9], params[10], params[11]] # proportional [x,y,z]
    Kd_ang = [params[12], params[13], params[14]]  # derivative [x,y,z]
    Ki_ang = [params[15], params[16], params[17]] # integral [x,y,z]

    Ki_sat_ang = 0.1*np.ones(3)  # saturation for integral controller (prevent windup) [x,y,z]

    # Create quadcotper with position and angle controller objects
    quadcopter = Quadcopter(pos,vel,ang,ang_vel,r_ref, 0.01,dL=params[18],render=render)
    pos_controller = PID_Controller(Kp_pos, Kd_pos, Ki_pos, Ki_sat_pos, dt)
    angle_controller = PID_Controller(Kp_ang, Kd_ang, Ki_ang, Ki_sat_ang, dt)

    max_angle = math.pi/12. #radians, max angle allowed at any time step

    # Initialize results arrays
    total_error = []
    position_total = []
    total_thrust = []

    def settling_time(tolerance=0.1):
        index = max([index for index,value in enumerate(total_error) if value > tolerance])
        return time_index[index]

    def initialize_results(res_array, num):
        for i in range(num):
            res_array.append([])


    position = []
    initialize_results(position,3)

    velocity = []
    initialize_results(velocity,3)

    angle = []
    initialize_results(angle,3)

    angle_vel = []
    initialize_results(angle_vel,3)

    motor_thrust = []
    initialize_results(motor_thrust,4)

    body_torque = []
    initialize_results(body_torque,3)


    # Simulation
    for time in enumerate(time_index):

        #find position and velocity error and call positional controller
        pos_error = quadcopter.calc_pos_error(quadcopter.pos)
        vel_error = quadcopter.calc_vel_error(quadcopter.vel)
        des_acc = pos_controller.control_update(pos_error,vel_error)
        #Modify z gain to include thrust required to hover
        des_acc[2] = (gravity + des_acc[2])/(math.cos(quadcopter.angle[0]) * math.cos(quadcopter.angle[1]))

        #calculate thrust needed
        thrust_needed = quadcopter.mass * des_acc[2]

        #Check if needed acceleration is not zero. if zero, set to one to prevent divide by zero below
        mag_acc = np.linalg.norm(des_acc)
        if mag_acc == 0:
            mag_acc = 1
        #print(-des_acc[1] / mag_acc / math.cos(quadcopter.angle[1]),des_acc[0] / mag_acc)
        ratio1=-des_acc[1] / mag_acc / math.cos(quadcopter.angle[1])
        ratio2=des_acc[0] / mag_acc
        if ratio1 < -1.0:
            print('ang_des[0]: asin argument less than -1')
            ratio1 = -1.0
        elif ratio1 > 1.0:
            print('ang_des[0]: asin argument bigger than 1')
            ratio1 = 1.0
        if ratio2 < -1.0:
            print('ang_des[1]: asin argument less than -1')
            ratio2 = -1.0
        elif ratio2 > 1.0:
            print('ang_des[1]: asin argument bigger than 1')
            ratio2 = 1.0

        #use desired acceleration to find desired angles since the quad can only move via changing angles
        #ang_des = [math.asin(-des_acc[1] / mag_acc / math.cos(quadcopter.angle[1])),
            #math.asin(des_acc[0] / mag_acc),
            #0]
        ang_des = [math.asin(ratio1),
            math.asin(ratio2),
            0]

        #check if exceeds max angle
        mag_angle_des = np.linalg.norm(ang_des)
        if mag_angle_des > quadcopter.max_angle:
            ang_des = (ang_des / mag_angle_des) * quadcopter.max_angle

        #call angle controller
        quadcopter.angle_ref = ang_des
        ang_error = quadcopter.calc_ang_error(quadcopter.angle)
        ang_vel_error = quadcopter.calc_ang_vel_error(quadcopter.ang_vel)
        tau_needed = angle_controller.control_update(ang_error, ang_vel_error)

        #Find motor speeds needed to achieve desired linear and angular accelerations
        quadcopter.des2speeds(thrust_needed, tau_needed)

        # Step in time and update quadcopter attributes
        quadcopter.step()

        # Record key attributes for plotting
        position_total.append(np.linalg.norm(quadcopter.pos))

        position[0].append(quadcopter.pos[0])
        position[1].append(quadcopter.pos[1])
        position[2].append(quadcopter.pos[2])

        velocity[0].append(quadcopter.vel[0])
        velocity[1].append(quadcopter.vel[1])
        velocity[2].append(quadcopter.vel[2])

        angle[0].append(np.rad2deg(quadcopter.angle[0]))
        angle[1].append(np.rad2deg(quadcopter.angle[1]))
        angle[2].append(np.rad2deg(quadcopter.angle[2]))

        angle_vel[0].append(np.rad2deg(quadcopter.ang_vel[0]))
        angle_vel[1].append(np.rad2deg(quadcopter.ang_vel[1]))
        angle_vel[2].append(np.rad2deg(quadcopter.ang_vel[2]))

        motor_thrust[0].append(quadcopter.speeds[0]*quadcopter.kt)
        motor_thrust[1].append(quadcopter.speeds[1]*quadcopter.kt)
        motor_thrust[2].append(quadcopter.speeds[2]*quadcopter.kt)
        motor_thrust[3].append(quadcopter.speeds[3]*quadcopter.kt)

        body_torque[0].append(quadcopter.tau[0])
        body_torque[1].append(quadcopter.tau[1])
        body_torque[2].append(quadcopter.tau[2])

        total_thrust.append(quadcopter.kt * np.sum(quadcopter.speeds))

        # Positional error
        r_error = quadcopter.pos_ref - quadcopter.pos
        total_error.append(np.linalg.norm(r_error))
    obj_val = np.abs(settling_time()-target)
    #obj_val = settling_time()
    print('Ixx = ',quadcopter.Ixx)
    print('Iyy = ',quadcopter.Iyy)
    print('Izz = ',quadcopter.Izz)
    print('obj_val:',obj_val)
    return [params, obj_val, ang_vel, total_error, position, velocity, angle, angle_vel, motor_thrust, body_torque, total_thrust, settling_time()]
#           [0]     [1]      [2]      [3]          [4]       [5]       [6]    [7]        [8]           [9]          [10]

def obj_fun1(params):
    res = obj_fun(params)
    return res[1]

def obj_fun2(params):
    res = obj_fun(params)
    return [res[0],res[1]]


# Write random values to screen
def write_init_ang_vel_to_screen():
    ''' 
    The program initializes with a random perturbation in angular velocity on the vehicle. 
    This simulates a wind disturbace.
    This is a display of the random disturbance
    '''
    print('Initial angular velocities (deg/s):')
    print(np.rad2deg(ang_vel_init))
    print('Total magnitude of angular velocity (deg/s)')
    print(np.linalg.norm(np.rad2deg(ang_vel_init)))

# Visualising the results
def obj_fun_hist_plot(res_his):
    #import pylab as plt
    keys=list(res_his.keys())
    plt.figure()
    for key in keys:
        plt.plot(res_his[key],label=key)
    plt.xlabel('iteration')
    plt.ylabel('Settling time difference from target [s]')
    plt.legend(loc='upper right')
    plt.savefig('objective_value_histories.pdf')
    #plt.show()
def error_plot(r,method):
    ''' Plots to the magnitude of the position error vector (m)'''
    plt.figure()
    plt.plot(time_index, r[3])
    plt.title('Quadcopter distance from reference point over time')
    plt.xlabel('time (s)')
    plt.ylabel('error (m)')
    plt.savefig('error_plot_%s.pdf'%method)
    #plt.show()

def simple_plot(r):
    ''' 
    Plots the laterial position, vertical position, and Euler angles over time.
    This is a quick plot for trouble shooting
    '''
    fig =  plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
    # Lateral position plots
    axes = fig.add_subplot(1, 3, 1)
    #axes.plot(time_index, position[0], label= 'x')
    #axes.plot(time_index, position[1], label= 'y')
    axes.plot(r[4][0], r[4][1])
    axes.set_title('Lateral Postion Over Time')
    axes.set_xlabel('x-position (m)')
    axes.set_ylabel('y-position (m)')
    axes.legend()

    # Vertical position plot
    axes = fig.add_subplot(1, 3, 2)
    axes.plot(time_index, r[4][2], label= 'z')
    axes.set_title('Vertical Position Over Time')
    axes.set_xlabel('time (s)')
    axes.set_ylabel('altitude (m)')

    # Angles over time
    axes = fig.add_subplot(1, 3, 3)
    axes.plot(time_index, r[6][0], label= 'phi')
    axes.plot(time_index, r[6][1], label= 'theta')
    axes.plot(time_index, r[6][2], label= 'psi')
    axes.set_title('Euler Angles Over Time')
    axes.set_xlabel('time (s)')
    axes.set_ylabel('angle (deg)')
    axes.legend()

    plt.tight_layout(pad=0.4, w_pad=2.5, h_pad=2.0)
    #plt.show()


def total_plot(r,method):
    '''
    This is a full plot of the results. It will plot the 3D flight path, vertical and lateral positions,
    lateral velocity, motor thrusts, body torques, Euler angles, and angular velocity of the vehicle.
    '''

    fig =  plt.figure(num=None, figsize=(16, 8), dpi=80, facecolor='w', edgecolor='k')

    # 3D Flight path
    axes = fig.add_subplot(2, 4, 1, projection='3d')
    axes.plot(r[4][0], r[4][1], r[4][2])
    axes.set_title('Flight Path')
    axes.set_xlabel('x (m)')
    axes.set_ylabel('y (m)')
    axes.set_zlabel('z (m)')

    # Lateral position plots
    axes = fig.add_subplot(2, 4, 2)
    axes.plot(time_index, r[4][0], label= 'x')
    axes.plot(time_index, r[4][1], label= 'y')
    axes.set_title('Lateral Postion')
    axes.set_xlabel('time (s)')
    axes.set_ylabel('position (m)')
    axes.legend()

    # Vertical position plot
    axes = fig.add_subplot(2, 4, 3)
    axes.plot(time_index, r[4][2], label= 'z')
    axes.set_title('Vertical Position')
    axes.set_xlabel('time (s)')
    axes.set_ylabel('altitude (m)')

    # Lateral velocity plots
    axes = fig.add_subplot(2, 4, 4)
    axes.plot(time_index, r[5][0], label= 'd(x)/dt')
    axes.plot(time_index, r[5][1], label= 'd(y)/dt')
    axes.plot(time_index, r[5][2], label= 'd(z)/dt')
    axes.set_title('Linear Velocity')
    axes.set_xlabel('time (s)')
    axes.set_ylabel('velocity (m/s)')
    axes.legend()

    # Motor speed plots
    axes = fig.add_subplot(2, 4, 5)
    axes.plot(time_index, r[8][0], label= 'motor 1')
    axes.plot(time_index, r[8][1], label= 'motor 2')
    axes.plot(time_index, r[8][2], label= 'motor 3')
    axes.plot(time_index, r[8][3], label= 'motor 4')
    axes.set_title('Motor Thrust')
    axes.set_xlabel('time (s)')
    axes.set_ylabel('Motor Thrust (N)')
    axes.legend()

    # Body torque over time
    axes = fig.add_subplot(2, 4, 6)
    axes.plot(time_index, r[9][0], label= 'x')
    axes.plot(time_index, r[9][1], label= 'y')
    axes.plot(time_index, r[9][2], label= 'z')
    axes.set_title('Body Torque')
    axes.set_xlabel('time (s)')
    axes.set_ylabel('torque (n-m)')
    axes.legend()

    # Angles over time
    axes = fig.add_subplot(2, 4, 7)
    axes.plot(time_index, r[6][0], label= 'phi')
    axes.plot(time_index, r[6][1], label= 'theta')
    axes.plot(time_index, r[6][2], label= 'psi')
    axes.set_title('Euler Angles')
    axes.set_xlabel('time (s)')
    axes.set_ylabel('angle (deg)')
    axes.legend()

    # Angular velocity over time
    axes = fig.add_subplot(2, 4, 8)
    axes.plot(time_index, r[7][0], label= 'd(phi)/dt')
    axes.plot(time_index, r[7][1], label= 'd(theta)/dt')
    axes.plot(time_index, r[7][2], label= 'd(psi)/dt')
    axes.set_title('Angular Velocity')
    axes.set_xlabel('time (s)')
    axes.set_ylabel('angular velocity (deg/s)')
    axes.legend()


    plt.tight_layout(pad=0.4, w_pad=2.5, h_pad=2.0)
    plt.savefig('total_plot_%s.pdf'%method)
    #plt.show()
params0 = [100.95, 100.95, 656.6, \
          20.8, 20.8, 15., \
          1.2, 1.2, 1.0, \
          56.9, 56.9, 25., \
          7.7, 7.7, 9., \
          0.5, 0.5, 0.1,0.0]
bounds = [(50,200),(50,200),(300,900),\
          (10,30),(10,30),(10,100), \
          (0.5,3.0),(0.5,3.0),(0.5,10.0), \
          (30,90),(30,90),(30,90), \
          (3,12),(3,12),(3,12), \
          (0.25,2.0),(0.25,2.0),(0.25,2.0),(-0.1,0.1)]
if __name__ == "__main__":
    methods =['DE']
    params_dic={}; res_his={}
    for method in methods: #'Powell', 'Nelder-Mead', 'COBYLA', 'SPSA', 'EGO', 'CMA-ES', 'DE', 'PSO'
        if method in ['Powell','Nelder-Mead','COBYLA']:
            res=scipy.optimize.minimize(obj_fun1, params0, args=(), method=method, jac=None, hess=None, hessp=None, bounds=bounds, constraints=(), tol=None, callback=None, options={'maxiter': 3})
            params_dic[method] = res.x
        elif method == 'DE':
            res=opal.de(obj_fun2, len(params0), X=None, bounds=bounds, gaptol=1.0e-06, pop=30, Cr=0.65, F=0.75, dither=True, IniNfeval=0, maxiter=5000, endMinf=1e-6, endMaxf=1e-6, maxNfeval=3000, args=())
            params_dic[method]=res[0]
            print('history: ',res[5])
            res_his[method]=np.array(res[5])[:,1]
        elif method == 'DE_warm':
            a=np.zeros([1,len(params0)])
            b=np.random.normal(size=[29,len(params0)])
            X=np.concatenate((a,b),axis=0)+params0
            res=opal.de(obj_fun2, len(params0), X=X, bounds=bounds, gaptol=1.0e-06, pop=30, Cr=0.65, F=0.75, dither=True, IniNfeval=0, maxiter=5000, endMinf=1e-6, endMaxf=1e-6, maxNfeval=120, args=())
            params_dic[method]=res[0]
            res_his[method]=np.array(res[5])[:,1]
    print('initial params: ',params0)
    print('optimal params: ',params_dic)
    r = obj_fun(params0, render=True)
    for method in list(params_dic.keys()):
        r = obj_fun(params_dic[method], render=True)
        print ('settling time (%s):'%method,r[11])
        error_plot(r,method)
        #simple_plot(r)
        total_plot(r,method)
    obj_fun_hist_plot(res_his)
    ang_vel_init = r[2].copy()  #record for later display

    write_init_ang_vel_to_screen()

    plt.show()
