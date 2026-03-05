# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from pyworld3 import World3
from pyworld3.utils import plot_world_variables

params = {"lines.linewidth": "3"}
plt.rcParams.update(params)

def fioac_contol(t, world, k):
    return 0.45

def isopc_control(t, world, k):
    if t<=2000:
        return 1
    return 1.2

def icor_control(t, world, k):
    if t <= 2023:
        return world.icor[k]
    else:
        return world.fioac[k]
    
def ifpc_control(t, world, k):
    return 1.2


world3 = World3(year_max=2100)
world3.set_world3_control(fioac_control=fioac_contol, icor_control=icor_control, isopc_control=isopc_control, ifpc_control=ifpc_control)
world3.init_world3_constants()
world3.init_world3_variables()
world3.set_world3_table_functions()
world3.set_world3_delay_functions()
world3.run_world3(fast=False)



plot_world_variables(
    world3.time,
    [world3.al, world3.pal, world3.uil, world3.lfert],
    ["AL", "PAL", "UIL", "LFERT"],
    [[0, 25e8], [0, 23e8], [0, 120e6], [0, 800]],
    figsize=(7, 5),
    title="World3 control run - Agriculture",
)
plt.grid()


plot_world_variables(
    world3.time,
    [world3.ic, world3.sc],
    ["IC", "SC"],
    [[0, 12e12], [0, 50e11]],
    figsize=(7, 5),
    title="World3 control run - Capital",
)
plt.grid()

plot_world_variables(
    world3.time,
    [world3.ppolx],
    ["PPOL"],
    [[0, 20e8]],
    figsize=(7, 5),
    title="World3 control run - Pollution",
)
plt.grid()

plot_world_variables(
    world3.time,
    [world3.p1, world3.p2, world3.p3, world3.p4],
    ["P1", "P2", "P3", "P4"],
    [[0, 3e9], [0, 5e9], [0, 2e9], [0, 2e9]],
    figsize=(7, 5),
    title="World3 control run - Population",
)
plt.grid()

plot_world_variables(
    world3.time,
    [world3.nr],
    ["NR"],
    [[0, 1e12]],
    figsize=(7, 5),
    title="World3 control run - Resources",
)
plt.grid()
plt.show()

"""

plot_world_variables(
    world3.time,
    [world3.nrfr, world3.iopc, world3.fpc, world3.pop, world3.ppolx],
    ["NRFR", "IOPC", "FPC", "POP", "PPOLX"],
    [[0, 1], [0, 1e3], [0, 1e3], [0, 16e9], [0, 32]],
    img_background="./img/fig7-7.png",
    figsize=(7, 5),
    title="World3 control run - General",
)
plt.savefig("fig_world3_control_general.pdf")




plot_world_variables(
    world3.time,
    [
        world3.fcaor,
        world3.io,
        world3.tai,
        world3.aiph,
        world3.fioaa,
        world3.fioac_control_values,
    ],
    ["FCAOR", "IO", "TAI", "AI", "FIOAA"],
    [[0, 1], [0, 4e12], [0, 4e12], [0, 2e2]],
    img_background="./img/fig7-8.png",
    figsize=(7, 5),
    title="World3 control run - Capital sector",
)
plt.savefig("fig_world3_control_capital.pdf")

plot_world_variables(
    world3.time,
    [world3.ly, world3.al, world3.fpc, world3.lmf, world3.pop],
    ["LY", "AL", "FPC", "LMF", "POP"],
    [[0, 4e3], [0, 4e9], [0, 8e2], [0, 1.6], [0, 16e9]],
    img_background="./img/fig7-9.png",
    figsize=(7, 5),
    title="World3 control run - Agriculture sector",
)
plt.savefig("fig_world3_control_agriculture.pdf")


"""






