# File to store reward functions

import numpy as np

def reward_hwi(world, k=None):
    """
    Input:
        world   -   the World3 object to get reward from
        k       -   int representing a time step (optional)
    Output:
        if k=None, a vector with hwi values; otherwise the hwi at step k
    """
    reward = world.hwi
    if k is None:
        return reward
    else:
        return reward[k]

def reward_inv_ef(world, k=None):
    # Intended to minimise ecological footprint by maximising the additive inverse of the ecological footprint
    reward = -world.ef
    if k is None:
        return reward
    else:
        return reward[k]

def reward_ddiff(world, k=None, hwi_weight=1.0, ef_weight=0.25):
    # Inspired by doughnut economics
    reward = hwi_weight * world.hwi - ef_weight * world.ef
    #reward += 100
    if k is None:
        return reward
    else:
        return reward[k]

# TODO: find good values of constants, especially hwi_limit (currently it is a purely arbitrary value)
# TODO: think about other variables to bake into this reward
def reward_doughnut(world, k=None, hwi_weight=1.0, ef_weight=1.0, hwi_limit=0.55, hwi_alpha=4.0, hwi_beta=0.1, hwi_h=1.1, ef_limit=1.1, ef_alpha=0.5, ef_beta=-0.5, ef_h=0.6):
    # Doughnut economics approach which strongly punishes boundary transgression
    n = world.n
    ef_impact = ef_h*np.ones(n) - np.exp(ef_alpha*(world.ef-ef_limit*np.ones(n))+ef_beta)
    hwi_impact = hwi_h*np.ones(n) - np.exp(hwi_alpha*(hwi_limit*np.ones(n)-world.hwi)+hwi_beta)
    reward = hwi_weight*hwi_impact + ef_weight*ef_impact
    #reward = hwi_weight * world.hwi + ef_weight * world.ef - hwi_punish * np.exp(hwi_exp * (hwi_limit * np.ones(n) - world.hwi)) - ef_punish * np.exp(ef_exp * (world.ef - ef_limit * np.ones(n)))
    #reward += 100000 * np.ones(n)
    if k is None:
        return reward
    else:
        return reward[k]

def reward_doughnut2(world, k=None, hwi_weight=1.0, ef_weight=0.25, hwi_limit=0.55, hwi_punish=10, ef_limit=1.1, ef_punish=2.5):
    n = world.n

    hwi_matrix = np.zeros((2, n))
    hwi_matrix[1, :] = hwi_limit * np.ones(n) - world.hwi

    ef_matrix = np.zeros((2, n))
    ef_matrix[1, :] = world.ef - ef_limit * np.ones(n)

    reward = hwi_weight * world.hwi - ef_weight * world.ef - hwi_punish * np.max(hwi_matrix, axis=0) - ef_punish * np.max(ef_matrix, axis=0)

    if k is None:
        return reward
    else:
        return reward[k]

# TODO: Update HSDI reward (based on new variables available in World3-03)
def reward_HSDI_ref(world, world_reference, k=None):
    # HSDI
    # world - current World3 object
    # world_reference - standard run or World3 object to compare world to
    # k (optional) - time step

    # le
    min_le = np.min(world_reference.le) * 0.95
    max_le = np.max(world_reference.le) * 1.05
    I_le = (world.le - min_le) / (max_le - min_le)
    I_le = np.clip(I_le, 0, 1)

    # j/pop
    ref_jpop = world_reference.j / world_reference.pop
    min_jpop = np.min(ref_jpop) * 0.95
    max_jpop = np.max(ref_jpop) * 1.05
    jpop = world.j/world.pop
    I_jpop = (jpop - min_jpop) / (max_jpop - min_jpop)
    I_jpop = np.clip(I_jpop, 0, 1)

    # GDP
    min_gdp = np.min(world_reference.iopc + world_reference.sopc) * 0.95
    max_gdp = np.max(world_reference.iopc + world_reference.sopc) * 1.05
    world_gdp = world.iopc + world.sopc
    I_gdp = (world_gdp - min_gdp) / (max_gdp - min_gdp)
    I_gdp = np.clip(I_gdp, 0, 1)

    # Pollution
    min_pp_pop = np.min(world_reference.pp / world_reference.pop) * 0.95
    max_pp_pop = np.max(world_reference.pp / world_reference.pop) * 1.05
    pp_pop = world.pp / world.pop
    I_pp_pop = 1 - ((pp_pop - min_pp_pop) / (max_pp_pop - min_pp_pop))
    I_pp_pop = np.clip(I_pp_pop, 0, 1)

    # HSDI
    reward = (I_le * I_jpop * I_gdp * I_pp_pop) ** (1/4)

    if k is None:
        return reward
    else:
        return reward[k]