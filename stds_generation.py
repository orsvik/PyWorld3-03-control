# File to generate standard deviations for noise in World3
# Important note: we should use the same dt (and preferably n) here as the world instance we intend to use noise for

import numpy as np
from pyworld3 import World3
import pandas as pd
from tqdm import tqdm
import os
import json

# Declare constants
MIN_YEAR = 1900
MAX_YEAR = 2100
AGR_VARS = {"ly": 1.0}
POP_VARS = {"b": 0.5, "d1": 0.5, "d2": 0.5, "d3": 0.5, "d4": 0.5} # variable names and scaling of std in Population sector
CAP_VARS = {"j": 1e-5}
POL_VARS = {}
RES_VARS = {}

def do_standard_run(max_year=MAX_YEAR):
    world_standard = World3(year_max=max_year)
    world_standard.set_world3_control()
    world_standard.init_world3_constants()
    world_standard.init_world3_variables()
    world_standard.set_world3_table_functions()
    world_standard.set_world3_noise_stds()
    world_standard.set_world3_delay_functions()
    world_standard.run_world3(fast=False)
    return world_standard

def get_mu_sigma(world, variable, scale=1.0):
    data = getattr(world, variable)
    sigma = np.std(data) * scale
    mu = np.mean(data)
    return mu, sigma

def add_std_data(data, world, sect_vars, sect_name):
    for var in sect_vars.keys():
        _, std = get_mu_sigma(world, var, scale=sect_vars[var])
        data.append([sect_name, var, std])

def write_to_json(json_file, data):
    rows = np.shape(data)[0]
    json_str = "[\n"
    for i in range(rows):
        item = data[i]
        sector, var_name, noise_std = item[0], item[1], item[2]
        new_data_point = {
            "sector": sector,
            "var_name": var_name,
            "noise_std": noise_std
        }
        json_str += json.dumps(new_data_point, indent=4)
        if i == rows-1:
            json_str += "\n"
        else:
            json_str += ",\n"
    json_str += "]"
    with open(json_file, "w") as njson:
        njson.write(json_str)

# Main:
def generate_stds():
    world_standard = do_standard_run()
    data = []
    add_std_data(data, world=world_standard, sect_vars=AGR_VARS, sect_name="Agriculture")
    add_std_data(data, world=world_standard, sect_vars=POP_VARS, sect_name="Population")
    add_std_data(data, world=world_standard, sect_vars=CAP_VARS, sect_name="Capital")
    json_file = "pyworld3/noise_stds.json"
    add_std_data(data, world=world_standard, sect_vars=POL_VARS, sect_name="Pollution")
    add_std_data(data, world=world_standard, sect_vars=RES_VARS, sect_name="Resource")
    json_file = os.path.join(os.path.dirname(__file__), json_file)
    write_to_json(json_file, data)
