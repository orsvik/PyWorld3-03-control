"""
New version of data_generation.py written by students Emil Johansson and Linnéa Bäckvall last year
"""

import numpy as np
from pyworld3 import World3
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

state_variables = ["p1", "p2", "p3", "p4", "ic", "sc", "nr", "al", "pal", "uil", "lfert", "pcrum", "time"] # state variables in World3-03

# Standard run used for randomizing initial state
world_standard = World3(year_max=2100)
world_standard.set_world3_control()
world_standard.init_world3_constants()
world_standard.init_world3_variables()
world_standard.set_world3_table_functions()
world_standard.set_world3_delay_functions()
world_standard.run_world3(fast=False)

def J_func(reward):
    iterations = reward.shape[0]
    J = np.zeros((iterations, 1))
    J[iterations-1] = reward[iterations-1]
    for k in range(2, iterations+1):
        J[iterations-k] = reward[iterations-k] + J[iterations-k+1]
    return J

def reward_hwi(world):
    return world.hwi

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
    not_time_variables = [var for var in state_variables if (var != 'time' and var != 'pcrum')]
    initial_values = generate_initial(runs, not_time_variables)

    df_list = []

    for run in tqdm(range(runs)):
        MIN_YEAR = 1900
        MAX_YEAR = 2100
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
    df = main_loop(chosen_reward, 10) # use 1 for now to test, limit time
    df.to_parquet(f"datasets/data_{reward_func_name}.parquet", index=False)

main(reward_hwi)