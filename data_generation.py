"""
Generate training data to be used in RL file and save the data as a dataframe with states and rewards as a parquet file

Authors of this modified file: Evelina Örsvik and Linnea Stålberg

Date of modifications: March-April 2026

Note on original authors: This file is an adaptation of Emil Johansson's and Linnéa Bäckvall's training data file `data_generation.py` available at https://github.com/emilj610/pyworld3A3. It has been modified to fit our (the new authors') purposes.
"""

# File naming convention: inttrainingset_data_rewardname.parquet
# Path: datasets/traintest/rewardname

# Imports
import numpy as np
from pyworld3 import World3
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import time
import json
import os

from rewards import *
from pyworld3.utils import standard_setup

# Declare state variables of different categories
state_variables = ["p1", "p2", "p3", "p4", "ic", "sc", "al", "pal", "uil", "lfert", "pp", "nr", "time"] # state variables in PyWorld3-03
no_init_vars = ["time"] # state variables in PyWorld3-03 not included in init_world3_constants
init_vars = [var for var in state_variables if (var not in no_init_vars)] # state variables in PyWorld3-03 that ARE included in init_world3_constants

# Declare constants used throughout the file
MIN_YEAR = 1900
MAX_YEAR = 2100
PLOT = False # toggle plots and prints
DEBUG_MODE = False # toggle debug mode, data does not get saved to file (to prevent overwriting better/useful data that may have taken a long time to generate)
FAST = True
NOISE = False
RUNS = 10

ID = "1" # training set ID

# Standard run, used for randomising initial state
world_standard = World3(year_max=MAX_YEAR, noise=NOISE)
standard_setup(world_standard)
world_standard.run_world3(fast=False)

def J_func(reward):
    """
    In:
        reward - numpy array: rewards (g values) for the simulation
    Out:
        Array of J function values, where J is the cumulative reward for each step onwards
    """
    iterations = reward.shape[0]
    J = np.zeros((iterations, 1))
    J[iterations-1] = reward[iterations-1] # the last J value is simply the last g value
    for k in range(2, iterations+1):
        # J[n] is the reward at step n plus J[n+1]
        J[iterations-k] = reward[iterations-k] + J[iterations-k+1]
    return J


# -- REWARD FUNCTIONS DEFINITION ZONE (for those who must be modified in some way) --

def reward_HSDI(world):
    return reward_HSDI_ref(world, world_standard)


# --

# Select here which reward function to use throughout the rest of the file
reward_dict = {"HSDI" : reward_HSDI,
               "HWI" : reward_hwi,
               "ddiff" : reward_ddiff,
               "doughnut" : reward_doughnut,
               "doughnut2" : reward_doughnut2,
               "inv_ef" : reward_inv_ef}

REWARD_NAME = "doughnut"
REWARD_FUNC = reward_dict[REWARD_NAME]

if PLOT:
    # Plots and prints for intuition
    print(REWARD_FUNC(world_standard))
    plt.plot(REWARD_FUNC(world_standard))
    plt.show()

def get_mu_sigma(world, variable):
    """
    In:
        world - World3 object: the current/relevant world
        variable - str: current variable
    Out:
        the mean and half of the standard deviation of the variable's data points over the whole run of the World3 object world
    """
    data = getattr(world, variable)
    mean = data[0] # CHANGE TO data.mean()??
    std = np.std(data) / 2 # regularisation, prevent extreme values
    return mean, std

def generate_initial(total_runs, variables):
    """
    In:
        total_runs - int: total number of simulations to generate initial data for
        variables - list[String]: the variables that will be initialised randomly with Gaussian distribution
    Out:
        initial_variables - list[dictionary<String,float>]: Dictionary with the initial variables (name and value)
    
    Generate initial values from a Gaussian distribution with mean and variance decided by each variable's values over the standard run
    """
    array = []
    for _ in range(total_runs):
        dict = {}
        for variable in variables:
            mu, sigma = get_mu_sigma(world_standard, variable)
            value = np.random.normal(mu, sigma)
            while value <= 0:
                value = np.random.normal(mu, sigma)
            dict[variable+"i"]=value
        array.append(dict)
    return array

def write_to_json(num_runs, fast_TF, noise_TF, seed_gen=0, seed_lst=[], reward_name=REWARD_NAME, id=ID):
    # json_file_dir should be datasets/traintest/rewardname/ID_data_rwrdname.json
    # Must create such file before running code, so the path exists
    json_file_dir = f"datasets/traintest/{reward_name}/{id}_data_{reward_name}.json"
    json_str = "[\n"
    if noise_TF:
        data_point = {
            "num_runs": num_runs,
            "fast_TF": fast_TF,
            "noise_TF": noise_TF,
            "seed_gen": seed_gen,
            "seed_lst": seed_lst
        }
    else:
        data_point = {
            "num_runs": num_runs,
            "fast_TF": fast_TF,
            "noise_TF": noise_TF
        }
    json_str += json.dumps(data_point, indent=4)
    json_str += "\n]"
    json_file = os.path.join(os.path.dirname(__file__), json_file_dir)
    with open(json_file, "w") as njson:
        njson.write(json_str)

def main_loop(reward_func, runs=100):
    """
    In:
        reward func - function: function that takes a World3 object as input and returns an array of rewards
        runs - int: number of runs
    Returns:
        dataframe with initial states and one cumulative reward J for each World3 instance (selected start year, end year, and initial values; no controls)
    
    Simulates runs of the World3 model without any control functions. The input is randomised with values based on the standard run.
    25% of the runs start at a random year selected uniformly between MIN_YEAR + 1 and MAX_YEAR, and end at MAX_YEAR
    25% of the runs end at a random year selected uniformly between MIN_YEAR + 1 and MAX_YEAR, and start at MIN_YEAR
    The rest of the runs start at MIN_YEAR and end at MAX_YEAR.
    MIN_YEAR and MAX_YEAR are defined in the beginning of this file
    """
    variables = state_variables
    initial_values = generate_initial(runs, init_vars) # init_vars defined in the beginning of this file

    df_list = []

    for run in tqdm(range(runs)):
        # Adapt start and end year
        if run > 0.75 * runs:
            min_year = np.random.randint(MIN_YEAR + 1, MAX_YEAR)
            max_year = MAX_YEAR
        elif run < 0.25 * runs:
            max_year = np.random.randint(MIN_YEAR + 1, MAX_YEAR)
            min_year = MIN_YEAR
        else:
            min_year = MIN_YEAR
            max_year = MAX_YEAR

        # Run model without controls but with selected start year, end year, and initial values
        world3 = World3(year_max=MAX_YEAR, year_min=min_year, noise=NOISE)
        world3.set_world3_control()
        world3.init_world3_constants(**initial_values[run])
        world3.init_world3_variables()
        world3.set_world3_table_functions()
        world3.set_world3_noise_stds()
        world3.set_world3_delay_functions()
        world3.run_world3(fast=FAST)

        # Save data (of this specific run) to dataframe in columns named after the variables, and the reward in a column named "J". The cumulative reward is saved for each time step in the run.
        run_df = pd.DataFrame({var: getattr(world3, var) for var in variables})
        run_df["J"] = J_func(reward_func(world3))
        run_df = run_df[run_df['time'] <= max_year] # clear dataframe of data where the it does not hold that the time is less than or equal to max_year
        df_list.append(run_df) # append to list of all run dataframes

    # Collect all run dataframes into one common dataframe
    df = pd.concat(df_list, ignore_index=True) # for why ignore_index=True, see pandas.concat documentation
    return df

def main(chosen_reward=REWARD_FUNC, reward_name=REWARD_NAME):
    if DEBUG_MODE:
        print("Debug mode active. Toggle by selecting DEBUG_MODE=False in the code and restarting the Python run.")
    #reward_func_name = chosen_reward.__name__
    print(f"Creating dataset for {reward_name}")
    df = main_loop(chosen_reward, runs=RUNS) # use small number to test, limit time; 1000 was used in BT 2025
    if DEBUG_MODE:
        print("Debug mode. Data does not get saved to file.")
    else:
        df.to_parquet(f"datasets/traintest/{reward_name}/{ID}_data_{reward_name}.parquet", index=False) # see pandas.DataFrame.to_parquet documentation for why index=False
        write_to_json(RUNS, FAST, NOISE, reward_name=reward_name, id=ID) # TODO: seed_gen, seed_lst
        print("The data was saved to file.")

main(REWARD_FUNC, REWARD_NAME)