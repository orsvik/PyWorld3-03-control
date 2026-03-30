# Playground to test different reward functions

from pyworld3 import World3
from rewards import *
from pyworld3.utils import standard_setup
from matplotlib import pyplot as plt

sr = World3(year_max=2100)
standard_setup(sr)
sr.run_world3(fast=True)

plt.plot(reward_doughnut(sr))
plt.show()

plt.plot(sr.ef)
plt.show()