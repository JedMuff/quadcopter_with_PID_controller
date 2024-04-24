from engine import MujocoSim

import mujoco
import numpy as np

def get_quaternion_from_euler(roll, pitch, yaw):
  """
  Convert an Euler angle to a quaternion.
   
  Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.
 
  Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
  """
  qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
  qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
  qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
 
  return [qx, qy, qz, qw]

def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = np.arcsin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)

        return [roll_x, pitch_y, yaw_z] # in radians

class Quadcopter():

    def __init__(self, pos, vel, angle, ang_vel, pos_ref, dt):

        self.sim = MujocoSim("model")

        L = 0.2 # length from body center to prop center, m
        gear = 0.0201
        morph_params = {
            'num_motors' : 4, # number of motors on the vehicle
            'motor_gear' : [gear, -gear, gear, -gear],
            'motor_trans' : [[  L,  L, 0.0 ],
                            [  L, -L, 0.0 ],
                            [ -L, -L, 0.0 ],
                            [ -L,  L, 0.0 ] ],
            'motor_quads' : [[ 1.0, 0.0, 0.0, 0.0],
                            [ 1.0, 0.0, 0.0, 0.0],
                            [ 1.0, 0.0, 0.0, 0.0],
                            [ 1.0, 0.0, 0.0, 0.0] ],
            'motor_names' : ["FR", "BR", "BL", "FL"],
            'motor_masses' : [0.005, 0.005, 0.005, 0.005],
            'core_mass' : 0.506, # total mass of the vehicle, kg
            'ixx' : 8.11858e-5,
            'iyy' : 8.11858e-5,
            'izz' : 6.12233e-5,
            'ixy' : 0.0,
            'ixz' : 0.0,
            'iyz' : 0.0,
            'goal_pos' : pos_ref
        }

        self.sim.build_sim(morph_params=morph_params, asarray=False)
        self.sim.sim_reset()
        self.sim.init_viewer()

        # Initial Sim variables
        self._pos = pos  # position of vehicle in inertial frame [x,y,z], meters
        self.sim.data.qpos[0:3]= pos
        self.vel = vel  #velocity of vehicle in inertial frame [x_dot, y_dot, z_dot], m/s
        self.angle = angle # orintation of vehicle inertial frame in radians [roll, pitch, yaw] -> [phi, theta, psi]
        self.ang_vel = ang_vel # angular velocity of inertial angles in rads/sec [phi_dot, theta_dot, psi_dot]
        self.lin_acc = np.array([0., 0., 0.]) # linear acceleration of vehicle in inertial frame [d^2(x)/dt, d^2(y)/dt, d^2(z)/dt], m/s^2
        self.ang_acc = np.array([0., 0., 0.]) #angular acceleration of vehicle in inertial frame [d^2(phi)/dt, d^2(theta)/dt, d^2(psi)/dt], rad/s^2

        # Desired reference states
        self.pos_ref = pos_ref #desired position [x, y, z] in inertial frame, m
        self.vel_ref = [0., 0., 0.]  #desired velocity [d(x)/dt, d(y)/dt, d(z)/dt] in inertial frame, m/s
        self.lin_acc_ref = [0., 0., 0.]  #desired acceleration [d^2(x)/dt, d^2(y)/dt, d^2(z)/dt] in inertial frame, m/s^2
        self.angle_ref = [0., 0., 0.] #desired angle [phi, theta, psi], radians
        self.ang_vel_ref = [0., 0., 0.,] #desired angular velocity [d(phi)/dt, d(theta)/dt, d(psi)/dt], radians/sec
        self.ang_acc_ref = [0., 0., 0.,] #desired angular acceleration [d^2(phi)/dt, d^2(theta)/dt, d^2(psi)/dt], radians/sec

        #Time measures
        self.dt = self.sim.dt
       
        #Environment variables
        gravity = 9.8 # acceleration due to gravity, m/s^2
        self.density = 1.225 # air density, kg/m^3

        # Vehicle constants
        self.num_motors = morph_params["num_motors"] # number of motors on the vehicle
        self.mass = morph_params["core_mass"] + np.sum(morph_params["motor_masses"]) # total mass of the vehicle, kg
        self.Ixx = morph_params["ixx"]  # mass-moment of inertial about x-axis, kg-m^2
        self.Iyy = morph_params["iyy"]  # mass-moment of inertial about y-axis, kg-m^2
        self.Izz = morph_params["izz"] # mass-moment of inertial about z-axis, kg-m^2
        self.A_ref = 0.02 # reference area for drag calcs, m^2 
        self.L = L
        
        self.kt = 1e-7 # proportionality constant to convert motor rotational speed into thrust (T=kt*omega^2), N/(rpm)^2
        self.b_prop = 1e-9 # proportionality constant to convert motor speed to torque (torque = b*omega^2), (N*m)/(rpm)^2
        self.Cd = 1 # drag coefficient
        self.thrust = self.mass * gravity
        self.speeds = np.ones(morph_params["num_motors"]) * ((morph_params["core_mass"] * gravity) / (self.kt * morph_params["num_motors"])) # initial speeds of motors
        self.tau = np.array([-9.72637046e-06, -4.86300202e-06,  0.00000000e+00]) # JM: Initial torque?

        self.maxT = 16.5 #  max thrust from any single motor, N
        self.minT = 0.5 # min thrust from any single motor, N 

        self.I = np.array([[self.Ixx, 0, 0],[0, self.Iyy, 0],[0, 0, self.Izz]])
        self.g = np.array([0, 0, -gravity])

        self.core_id = self.sim.get_id("core", "body")
        self.max_angle = np.pi/12 #radians, max angle allowed at any time step

    @property
    def pos(self):
        return self.sim.data.xpos[self.core_id]
    
    @pos.setter
    def pos(self, pos):
        self._pos = pos
        self.sim.data.qpos[0:3]= pos

    @property
    def vel(self):
        # return self._vel
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.sim.model, self.sim.data, mujoco.mjtObj.mjOBJ_BODY, self.core_id, vel, 0) # angular velocities, linear velocities 
        return vel[3:6]
    
    @vel.setter
    def vel(self, vel):
        self.sim.data.qvel[0:3]= vel

    @property
    def angle(self):
        q = self.sim.data.xquat[self.core_id]
        angle = euler_from_quaternion(q[1], q[2], q[3], q[0])
        return np.array(angle)
    
    @angle.setter
    def angle(self, angle):
        q = get_quaternion_from_euler(angle[0], angle[1], angle[2])
        self.sim.data.qpos[3:7]= [q[3], q[0], q[1], q[2]]

    @property
    def ang_vel(self):
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.sim.model, self.sim.data, mujoco.mjtObj.mjOBJ_BODY, self.core_id, vel, 0) # angular velocities, linear velocities 
        return vel[0:3]

    @ang_vel.setter
    def ang_vel(self, ang_vel):
        # q = get_quaternion_from_euler(ang_vel[0], ang_vel[1], ang_vel[2])
        self.sim.data.qvel[3:6]= ang_vel #[q[3], q[0], q[1], q[2]]

    def calc_pos_error(self, pos):
        ''' Returns the error between actual position and reference position'''
        pos_error = self.pos_ref - pos
        return pos_error
        
    def calc_vel_error(self, vel):
        ''' Returns the error between actual velocity and reference velocity'''
        vel_error = self.vel_ref - vel
        return vel_error

    def calc_ang_error(self, angle):
        ''' Returns the error between actual angle and reference angle'''
        angle_error = self.angle_ref - angle
        return angle_error

    def calc_ang_vel_error(self, ang_vel):
        ''' Returns the error between angular velocity position and reference angular velocity'''
        ang_vel_error = self.ang_vel_ref - ang_vel
        return ang_vel_error


    def body2inertial_rotation(self):
        ''' 
        Euler rotations from body-frame to global inertial frame
        angle 0 = roll (x-axis, phi)
        angle 1 = pitch (y-axis, theta)
        angle 2 = yaw (z-axis, psi)
        '''

        c1, c2, c3 = np.cos(self.angle[0]), np.cos(self.angle[1]), np.cos(self.angle[2])
        s1, s2, s3 = np.sin(self.angle[0]), np.sin(self.angle[1]), np.sin(self.angle[2])

        R = np.array([[c2*c3, c3*s1*s2 - c1*s3, s1*s3 + c1*s2*c3],
                     [c2*s3 , c1*c3 + s1*s2*s3, c1*s3*s2 - c3*s1],
                     [-s2   , c2*s1           , c1*c2           ]])
    
        return R


    def inertial2body_rotation(self):
        ''' 
        Euler rotations from inertial to body frame
            (Transpose of body-to-internal rotation)
        '''
        R = np.transpose(self.body2inertial_rotation())

        return R


    def thetadot2omega(self):
        '''rotate body angular velocity (Euler_dot) to inertial angular velocity (omega) '''

        R = np.array([[1, 0, -np.sin(self.angle[1])],
            [0,  np.cos(self.angle[0]), np.cos(self.angle[1])*np.sin(self.angle[0])],
            [0, -np.sin(self.angle[0]), np.cos(self.angle[1])*np.cos(self.angle[0])]])

        omega = np.matmul(R, self.ang_vel)
        return omega


    def omegadot2Edot(self,omega_dot):
        '''rotate inertial angular velocity (omega) to body angular velocity (Euler_dot) '''

        R = np.array([[1, np.sin(self.angle[0])*np.tan(self.angle[1]), np.cos(self.angle[0])*np.tan(self.angle[1])],
            [0, np.cos(self.angle[0]), -np.sin(self.angle[0])],
            [0, np.sin(self.angle[0])/np.cos(self.angle[1]), np.cos(self.angle[0])/np.cos(self.angle[1])]])

        E_dot = np.matmul(R, omega_dot)
        self.ang_acc = E_dot
        

    def find_omegadot(self, omega):
        ''' Find the angular acceleration in the inertial frame in rad/s '''
        omega = self.thetadot2omega() 
        
        # omega_dot = np.matmul(np.linalg.inv(self.I), (self.tau - np.cross(omega, np.matmul(self.I, omega))))
        omega_dot = np.linalg.inv(self.I).dot(self.tau - np.cross(omega, np.matmul(self.I, omega)))

        return omega_dot


    def find_lin_acc(self):
        ''' Find linear acceleration in m/s '''
        R_B2I = self.body2inertial_rotation()
        R_I2B = self.inertial2body_rotation()

        #body forces
        Thrust_body = np.array([0, 0, self.thrust])
        Thrust_inertial = np.matmul(R_B2I, Thrust_body) #convert to inertial frame

        vel_bodyframe = np.matmul(R_I2B, self.vel)
        drag_body = -self.Cd * 0.5 * self.density * self.A_ref * (vel_bodyframe)**2
        drag_inertial = np.matmul(R_B2I, drag_body)
        weight = self.mass * self.g

        acc_inertial = (Thrust_inertial + drag_inertial + weight) / self.mass
        self.lin_acc = acc_inertial


    def des2speeds(self,thrust_des, tau_des):
        ''' finds speeds of motors to achieve a desired thrust and torque '''

        # Needed torque on body
        e1 = tau_des[0] * self.Ixx * 10_000
        e2 = tau_des[1] * self.Iyy * 10_000
        e3 = tau_des[2] * self.Izz

        #less typing
        n = self.num_motors

        # Thrust desired converted into motor speeds
        weight_speed = thrust_des / (n*self.kt)

        # Thrust differene in each motor to achieve needed torque on body
        motor_speeds = []
        motor_speeds.append(weight_speed - (e2/((n/2)*self.kt*self.L)) - (e3/(n*self.b_prop)))
        motor_speeds.append(weight_speed - (e1/((n/2)*self.kt*self.L)) + (e3/(n*self.b_prop)))
        motor_speeds.append(weight_speed + (e2/((n/2)*self.kt*self.L)) - (e3/(n*self.b_prop)))
        motor_speeds.append(weight_speed + (e1/((n/2)*self.kt*self.L)) + (e3/(n*self.b_prop)))
        # Ensure that desired thrust is within overall min and max of all motors
        thrust_all = np.array(motor_speeds) * (self.kt)
        self.thrust_all = thrust_all
        over_max = np.argwhere(thrust_all > self.maxT)
        under_min = np.argwhere(thrust_all < self.minT)

        if over_max.size != 0:
            for i in range(over_max.size):
                motor_speeds[over_max[i][0]] = self.maxT / (self.kt)
        if under_min.size != 0:
            for i in range(under_min.size):
                motor_speeds[under_min[i][0]] = self.minT / (self.kt)
        
        self.speeds = motor_speeds

    def find_body_torque(self):
        tau = np.array([(self.L * self.kt * (self.speeds[3] - self.speeds[1])),
                        (self.L * self.kt * (self.speeds[2] - self.speeds[0])),
                        (self.b_prop * (-self.speeds[0] + self.speeds[1] - self.speeds[2] + self.speeds[3]))])
        
        self.tau = tau


    def step(self):
        #Thrust of motors
        self.thrust = self.kt * np.sum(self.speeds)

        tau = np.array(self.speeds)*self.b_prop

        self.sim.data.ctrl[:] = tau#self.thrust_all # Apply thrust
        #Linear and angular accelerations in inertial frame
        self.find_lin_acc()

        #torque on body from motor thrust differential
        self.find_body_torque()

        #Angular acceleration in inertial frame
        omega = self.thetadot2omega()  #angles in inertial frame
        omega_dot = self.find_omegadot(omega) #angular acceleration in inertial frame

        #Angular acceleration in body frame
        self.omegadot2Edot(omega_dot)
        
        # Update states based on time step
        ## Can comment this in and remove motor control above for debugging. This is how a step was done without Mujoco. ##
        # self.ang_vel += self.dt * self.ang_acc
        # self.angle += self.dt * self.ang_vel
        # self.vel += self.dt * self.lin_acc
        # self.pos = self.pos + self.dt * self.vel
        # self.sim.data.qpos[0:3]= self.pos
        self.sim.sim_step()
        self.sim.render_sim()

