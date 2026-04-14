# Help functions used in reinforcement_learning.ipynb and plotter.ipynb

def set_control_to_constant(control_name, value, world3):
    if control_name=="FIOAC":
        world3.fioac_control = lambda _: value # set the fioac control to constantly value
    elif control_name=="ISOPC":
        world3.isopc_control = lambda _: value # set the isopc control to constantly value
    elif control_name=="DCFSN":
        world3.dcfsn_control = lambda _: value
    elif control_name=="IOPC":
        world3.iopc_control = lambda _: value
    elif control_name=="ALIC":
        world3.alic_control = lambda _: value
    elif control_name=="FIOAA":
        world3.fioaa_control = lambda _: value
    elif control_name=="DPPOLX":
        world3.dppolx_control = lambda _: value
    elif control_name=="FIOAI":
        world3.fioai_control = lambda _: value
    elif control_name=="PPGF":
        world3.ppgf_control = lambda _: value

def set_control_value_list(control_name, value, world3, k):
    if control_name=="FIOAC":
        world3.fioac_control_values[k] = value
    elif control_name=="ISOPC":
        world3.isopc_control_values[k] = value
    elif control_name=="DCFSN":
        world3.dcfsn_control_values[k] = value
    elif control_name=="IOPC":
        world3.iopc_control_values[k] = value
    elif control_name=="ALIC":
        world3.alic_control_values[k] = value
    elif control_name=="FIOAA":
        world3.fioaa_control_values[k] = value
    elif control_name=="DPPOLX":
        world3.dppolx_control_values[k] = value
    elif control_name=="FIOAI":
        world3.fioai_control_values[k] = value
    elif control_name=="PPGF":
        world3.ppgf_control_values[k] = value

def get_default(control_name, world):
    if control_name=="FIOAC":
        return world.fioac[0]
    elif control_name=="ISOPC":
        return world.isopc[0]
    elif control_name=="DCFSN":
        return world.dcfsn[0]
    elif control_name=="IOPC":
        return world.iopc[0]
    elif control_name=="ALIC":
        return world.alic[0]
    elif control_name=="FIOAA":
        return world.fioaa[0]
    elif control_name=="DPPOLX":
        return world.dppolx[0]
    elif control_name=="FIOAI":
        return world.fioai[0]
    elif control_name=="PPGF":
        return world.ppgf[0]

def get_control_value_list(control_name, world):
    if control_name=="FIOAC":
        return world.fioac_control_values
    elif control_name=="ISOPC":
        return world.isopc_control_values
    elif control_name=="DCFSN":
        return world.dcfsn_control_values
    elif control_name=="IOPC":
        return world.iopc_control_values
    elif control_name=="ALIC":
        return world.alic_control_values
    elif control_name=="FIOAA":
        return world.fioaa_control_values
    elif control_name=="DPPOLX":
        return world.dppolx_control_values
    elif control_name=="FIOAI":
        return world.fioai_control_values
    elif control_name=="PPGF":
        return world.ppgf_control_values