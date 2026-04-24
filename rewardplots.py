from matplotlib import pyplot as plt
import numpy as np

def hwi_reward_demo(x, hwi_limit=0.55, hwi_alpha=4.0, hwi_beta=0.1, hwi_h=1.1):
    # Doughnut economics approach which strongly punishes boundary transgression
    return hwi_h - np.exp(hwi_alpha * (hwi_limit - x) + hwi_beta)

def ef_reward_demo(x, ef_limit=1.1, ef_alpha=0.5, ef_beta=-0.5, ef_h=0.6):
    return ef_h - np.exp(ef_alpha * (x - ef_limit) + ef_beta)

hwi_demo_xs = np.linspace(0, 1, 100)
hwi_demo_ys = hwi_reward_demo(hwi_demo_xs)

hwi_sig_vals_xs = np.array([0.550, 0.700, 0.800])
hwi_sig_vals_ys = hwi_reward_demo(hwi_sig_vals_xs)

plt.plot(hwi_demo_xs, hwi_demo_ys, label="HWI reward", color='r')
plt.plot(hwi_sig_vals_xs, hwi_sig_vals_ys, '*', color='k')
plt.grid(True)
plt.legend()
plt.xlabel("HWI")
plt.ylabel("HWI reward")
plt.show()

ef_demo_xs = np.linspace(0, 6.5, 650)
ef_demo_ys = ef_reward_demo(ef_demo_xs)

ef_sig_vals_xs = np.array([1.1, 4])
ef_sig_vals_ys = ef_reward_demo(ef_sig_vals_xs)

plt.plot(ef_demo_xs, ef_demo_ys, label="HEF reward", color='b')
plt.plot(ef_sig_vals_xs, ef_sig_vals_ys, '*', color='k')
plt.legend()
plt.grid(True)
plt.xlabel("HEF")
plt.ylabel("HEF reward")
plt.show()