import ast, re
import numpy as np

def find_and_remove(data, start, end):
    start_idx = data.find(start)
    end_idx = data.find(end)

    if end_idx == -1 or start_idx == -1:
        return 'None', 'None'

    variable_found = data[data.find(start)+len(start):end_idx]
    data = data[end_idx+1:]

    return data, variable_found

def find_and_remove_jinja_variable(data):
    data, key = find_and_remove(data, '{%set', '=')
    data, value = find_and_remove(data, 'default(', ')%')

    value_eval = ast.literal_eval(value)
    if value_eval == 'inf' or value_eval == '-inf': 
        value_eval = float(value_eval)

    return data, key, value_eval

def get_default_values(dir_file:str) -> dict:
    '''
    :param dir_file: the file to get the variables from
    :returns: dictionary of variable names and values in a file specified by the jinja syntax
    :rtype: dict
    '''
    dict = {}
    with open(dir_file, 'r') as file:
        data = file.read().replace('\n', '').replace(' ', '')

    data, key, value = find_and_remove_jinja_variable(data)
    while value != None and key != 'None':
        dict[key] = value
        data, key, value = find_and_remove_jinja_variable(data)

    return dict

# print(get_default_values('/home/Jed/Documents/uniwork/thesis/gym-adapt/gym_adapt/core/test.xml'))

def dict2nparray(dict):
    length = len(dict)
    keys = np.empty(length, dtype=object)
    values = np.empty(length, dtype=object)

    for i, (k, v) in enumerate(dict.items()):
        keys[i] = k
        values[i] = v

    return keys, values 

def nparray2dict(keys, values):
    dict = {}
    for idx in range(len(keys)):
        dict[keys[idx]] = values[idx]
    return dict

