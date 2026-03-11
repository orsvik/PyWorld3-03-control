"""
Generate training data to be used in RL file and save the data as a dataframe with states and rewards as a parquet file

Authors of this modified file: Evelina Örsvik and Linnea Stålberg

Date of modifications: March 2026

Note on original authors: This file is an adaptation of Emil Johansson's and Linnéa Bäckvall's training data file `data_generation.py` available at https://github.com/emilj610/pyworld3A3. It has been modified to fit our (the new authors') purposes.
"""

# Imports
import numpy as np
from pyworld3 import World3
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

# Declare state variables of different categories
state_variables = ["p1", "p2", "p3", "p4", "ic", "sc", "nr", "al", "pal", "uil", "lfert", "pcrum", "time"] # state variables in PyWorld3-03   # pcrum is currently a mystery
no_init_vars = ["pcrum", "time"] # state variables in PyWorld3-03 not included in init_world3_constants
init_vars = [var for var in state_variables if (var not in no_init_vars)] # state variables in PyWorld3-03 that ARE included in init_world3_constants

# Declare constants used throughout the file
MIN_YEAR = 1900
MAX_YEAR = 2100
PLOT = False # toggle plots and prints

# Standard run, used for randomising initial state
world_standard = World3(year_max=MAX_YEAR)
world_standard.set_world3_control()
world_standard.init_world3_constants()
world_standard.init_world3_variables()
world_standard.set_world3_table_functions()
world_standard.set_world3_delay_functions()
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
        J[iterations-k] = reward[iterations-k] + J[iterations-k+1]
    return J

def reward_hwi(world):
    return world.hwi

if PLOT:
    # Plots and prints for intuition
    print(reward_hwi(world_standard))
    plt.plot(reward_hwi(world_standard))
    plt.show()

def get_mu_sigma(world, variable):
    data = getattr(world, variable)
    mean = data[0]
    std = np.std(data) / 2 # normalization, done by last year's students
    return mean, std

def generate_initial(total_runs, variables):
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

def main_loop(reward_func, runs=100):
    variables = state_variables
    initial_values = generate_initial(runs, init_vars)

    df_list = []

    for run in tqdm(range(runs)):
        if run > 0.75 * runs:
            min_year = np.random.randint(MIN_YEAR + 1, MAX_YEAR)
            max_year = MAX_YEAR
        elif run < 0.25 * runs:
            max_year = np.random.randint(MIN_YEAR + 1, MAX_YEAR)
            min_year = MIN_YEAR
        else:
            min_year = MIN_YEAR
            max_year = MAX_YEAR
        world3 = World3(year_max=max_year, year_min=min_year)
        world3.set_world3_control()
        world3.init_world3_constants(**initial_values[run])
        world3.init_world3_variables()
        world3.set_world3_table_functions()
        world3.set_world3_delay_functions()
        world3.run_world3(fast=False) # fix fast=True at some point

        run_df = pd.DataFrame({var: getattr(world3, var) for var in variables})
        run_df["J"] = J_func(reward_func(world3))
        run_df = run_df[run_df['time'] <= max_year]
        df_list.append(run_df)

    df = pd.concat(df_list, ignore_index=True)
    return df

def main(chosen_reward):
    reward_func_name = chosen_reward.__name__
    print(f"Creating dataset for {reward_func_name}")
    df = main_loop(chosen_reward, 10) # use 10 for now to test, limit time
    df.to_parquet(f"datasets/data_{reward_func_name}.parquet", index=False)

main(reward_hwi)