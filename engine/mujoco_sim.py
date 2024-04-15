import glfw
import numpy as np
import mujoco as mj
from .mujoco_rendering import MujocoViewer
import mediapy

import warnings
from .morphological_file import MorphologicalFile

class MujocoSim(MorphologicalFile):
    '''This class serves as the bridge between the physics engine and python. It provides functions for initializing and getting information from the engine.

    :param morph: name of morphology
    :param engine: specifies what physics engine to use
    '''
    def __init__(self, env_dir_name) -> None:
        '''Constructor Method'''
        
        MorphologicalFile.__init__(self, env_dir_name)

        self.viewer = None
        self.recorder = None

        self.model = None
        self.build_sim()

        self.mj_objs = {'body': mj.mjtObj.mjOBJ_BODY, 'geom': mj.mjtObj.mjOBJ_GEOM, 'site': mj.mjtObj.mjOBJ_SITE, 
                        'camera': mj.mjtObj.mjOBJ_CAMERA, 'joint': mj.mjtObj.mjOBJ_JOINT, 'actuator':mj.mjtObj.mjOBJ_ACTUATOR}
        self.mj_jacs = {'body': mj.mj_jacBody, 'geom': mj.mj_jacGeom, 'site': mj.mj_jacSite}

        self.frame_skip = 1

    @property
    def dt(self):
        if self.model != None:
            dt = self.model.opt.timestep
        else:
            dt = None
        return dt

    @dt.setter
    def dt(self, dt):
        if self.model != None:
            self.model.opt.timestep = dt
        else:
            warnings.warn('dt not set because sim not built yet')
            

    def build_sim(self, morph_params=None, asarray=False) -> None:
        '''Used to initialize the simulation'''
        if self.model != None:
            dt = self.dt
        else:
            dt = None
            
        self.build_morphological_file(morph_params=morph_params, asarray=asarray)

        ASSETS=dict()
        self.model = mj.MjModel.from_xml_string(self.morph_file, ASSETS)
        self.data = mj.MjData(self.model)

        if dt != None:
            self.dt = dt

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

    def sim_reset(self) -> None:
        '''resets the simulation'''
        mj.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = 0
        mj.mj_forward(self.model, self.data)
        if self.recorder != None:
            self.recorder_reset()

    def sim_step(self) -> None:
        '''completes a simulation step '''
        mj.mj_step(self.model, self.data)
        mj.mj_forward(self.model, self.data)

    def init_recorder(self, camera=None):
        self.frames = []
        self.recorder = OffScreenViewer(self.model, self.data)
        self.sim_step()
        if camera != None:
            self.camera = camera
        else:
            self.camera = None

    def take_pic(self, camera):
        camera_id = self.get_id(camera, 'camera')
        pixels = self.recorder.render(render_mode=None, camera_id=camera_id, segmentation=False)
        return pixels

    def recorder_step(self):
        pixels = self.take_pic(self.camera)
        self.frames.append(pixels)
    
    def recorder_reset(self):
        self.frames = []
        del self.recorder
        self.init_recorder(camera=self.camera)
    
    def save_pic(self, pic, dir_name):
        mediapy.write_image(dir_name, pic)

    def save_video(self, dir_name, framerate=60):
        mediapy.write_video(dir_name, self.frames, fps=framerate)

    def send_command(self, ctrl) -> None:
        '''Commands the motors to move.
        
        :param ctrl: list of commands to send to the motors
        '''
        self.data.ctrl[:] = ctrl

    def init_viewer(self) -> None:
        '''initializes the viewer for rendering'''
        self.render_mode = 'human'
        self.viewer = MujocoViewer(self.model, self.data)

    def get_base_pos(self) -> list[float]:
        ''':returns: position of the robot'''
        return self.get_body_pos('robot')

    def get_id(self, name, obj_type):
        return mj.mj_name2id(self.model, self.mj_objs[obj_type], name)

    def get_geom_pos(self, name) -> list[float]:
        ''':returns: the position of a geom object type in mujoco.'''
        return self.data.geom_xpos[self.get_id(name, 'geom')]

    def get_body_pos(self, name) -> list[float]:
        ''':returns:  the position of a body object type in mujoco.'''
        return self.data.xpos[self.get_id(name, 'body')]

    def get_site_pos(self, name) -> list[float]:
        ''':returns: the position of a base object type in mujoco.'''
        return self.data.site_xpos[self.get_id(name, 'site')]

    def get_contact_pair(self, geom_name1, geom_name2):
        """Gets the pair ID of two objects that are in contact. 
        If the objects are not in contact it will return none.

        :param geom_name1: The geom name of the first object in enviroment
        :type geom_name1: str
        :param geom_name2: The geom name of the second object in enviroment
        :type geom_name2: str
        :return: The contact pair ID
        :rtype: int
        """
        # Find the proper geom ids
        geom_id1 = self.get_id(geom_name1, 'geom')
        geom_id2 = self.get_id(geom_name2, 'geom')

        contact = None

        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if (con.geom1 == geom_id1 and con.geom2 == geom_id2) or (con.geom2 == geom_id1 and con.geom1 == geom_id2):
                contact = con
                break

        return contact



    def render_sim(self) -> None:
        '''renders the environment, needs to be called on each step'''
        if self.viewer != None:
            self.viewer.render()

        if self.recorder != None:
            self.recorder_step()
            
    def close_viewer(self) -> None:
        '''destroys the viewer that renders the sim'''
        if self.viewer != None:
            self.viewer.close()
            self.viewer = None

    def do_simulation(self, a, frame_skip=1):
        '''
        Step the simulation n number of frames and applying a control action.
        '''

        for step in range(frame_skip):
            self.send_command(a)
            self.sim_step()

    def get_sim_state(self):
        """Return the position and velocity joint states of the model"""
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])

    def set_sim_state(self, qpos, qvel):
        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)
        if self.model.na == 0:
            self.data.act[:] = None
        mj.mj_forward(self.model, self.data)
