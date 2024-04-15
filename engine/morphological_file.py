from .template_renderer import TemplateRenderer
from .utils import *
import numpy as np
import os

class MorphologicalFile(object):
    '''This class contains how tools used to control the files morphological parameters'''
    
    def __init__(self, env_dir_name) -> None:
        
        self.renderer = TemplateRenderer(env_dir_name)
        self.renderer.resource_dirs[0]

        if any(fname.endswith('.xml') for fname in os.listdir(self.renderer.resource_dirs[0])):
            self.ext = '.xml'
        elif any(fname.endswith('.urdf') for fname in os.listdir(self.renderer.resource_dirs[0])):
            self.ext = '.urdf'
        
        self.morph_file_path =  self.renderer.resource_dirs[0]+'/generated'+self.ext
        self.default_parameters = get_default_values(self.renderer.resource_dirs[0]+'/const'+self.ext)
        self.parameters = self.default_parameters
        self.morph_params = self.parameters['morph_params']
        self.morph_limits = self.parameters['morph_limits']
        self.morph_groups = {}
        self.user_mp_map = {}

        # self.build_morphological_file() # sets self.morph_file
    
    def build_morphological_file(self, morph_params=None, asarray=False):

        if np.all(morph_params != None):
            self.update_morph_params(morph_params, asarray=asarray)
        
        self.morph_file = self.renderer.render_to_file('robot'+self.ext, self.morph_file_path, **self.parameters)
    
    def get_param(self, name, asarray=False):

        if asarray and bool(self.user_mp_map):
            params = self.parameters[name]
            params_subset = {k: params[k] for k in self.user_mp_map.keys()}
            _, v = dict2nparray(params_subset)
            return v

        elif asarray:
            k, v = dict2nparray(self.parameters[name])
            return v
    
        elif bool(self.user_mp_map):
            params = self.parameters[name]
            for group_name, associated in self.morph_groups.items():
                params[group_name] = params[associated[0]]
            params_subset = {v: params[v] for v in self.user_mp_map.values()}
            return params_subset

        return self.parameters[name]

    def get_morph_params(self, asarray=False) -> list:
        '''
        :return: list of morphology parameters and there current value
        :rtype: list 
        '''
        return self.get_param('morph_params', asarray=asarray)

    def get_morph_limits(self, asarray=False) -> list:
        limits = self.get_param('morph_limits', asarray=asarray)

        lower = np.empty(len(limits))
        upper = np.empty(len(limits))
        for i,l in enumerate(limits):
            lower[i] = l[0]
            upper[i] = l[1]

        return np.array([lower, upper])

    def get_user_map_data(self, params):
        d = {}
        for idx in self.user_mp_map.keys():
            d[self.user_mp_map[idx]] = params[idx]
        return d

    def update_params(self, name, params, asarray=False) -> list:
        if asarray and bool(self.user_mp_map):
            params = self.get_user_map_data(params)

        elif asarray:
            params = nparray2dict(list(self.parameters[name].keys()), params)

        params2change = params.keys()
        _params = self.parameters['morph_params']
        for group_name, mp_names in self.morph_groups.items():
            if group_name in params2change:
                val = params.pop(group_name)
                for param_name in mp_names:
                    _params[param_name] = val
        _params.update(params)
        self.parameters['morph_params'] = _params

    def update_morph_limits(self, limits, asarray=False) -> list:
        self.update_params('morph_limits', limits, asarray=asarray) 

    def update_morph_params(self, params, asarray=False):
        self.update_params('morph_params', params, asarray=asarray) 

    def get_morph_dims(self) -> int:
        '''
        :returns: how many morphological parameters there are
        :rtype: int
        '''
        return len(self.parameters['morph_params'])