import math
import numpy as np

# the benchmark objective functions we have been using
#

def rosenbrock_obj(params):
    z = 100*(params[1] - (params[0] ** 2) ** 2) + (params[0] - 1) ** 2
    z = 100*(math.pow(params[1] - math.pow(params[0], 2), 2)) + math.pow(params[0] - 1, 2)
    return -z   # flip to make this a maximization problem

def eggholder_obj(params):
    z = -(params[1] + 47) * np.sin(np.sqrt(np.abs(params[1] + params[0]/2 + 47))) - \
                    params[0]*np.sin(np.sqrt(np.abs(params[0] - (params[1] + 47))))
    return -z # to max instead of min

def cross_in_tray_obj(params):
    z = -0.0001 * math.pow(( np.abs(np.sin(params[0])*np.sin(params[1])* \
                                  sp.exp(np.abs(100 - np.sqrt(params[0]**2 + params[1]**2)/sp.pi)))+1),0.1)
    return -z # flip for maximization


