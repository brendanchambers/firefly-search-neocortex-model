import numpy as np
import json
from pprint import pprint



###### firefly config

config_prefix = 'check asyn measure 12-19-2016' # for making the filestring

verbose = 'test clean python architecture, asynchrony hacky' # optionally provide a general description of the current endeavor
PARAMS = ['p_ei','p_ie'] # ,'lognorm_sigma'] # name for easier printing
OBJECTIVES = ['asynchrony','stable duration'] # names for easier printing & reference
                    # (for now the second obj dimension is not necessary)

N_gen = 50
N_bugs = 10
N_params = len(PARAMS)
N_objectives = len(OBJECTIVES)

# range for [lognorm_sigma_manifold, p_ie]
MEANS = [0.2, 0.2] # ,-1] # for each param
STDS = [0.1, 0.1] # ,3]
MAXES = [0.4, 0.4] # , 5]
MINS = [0, 0] # , -10]  # sigma must be > 0

characteristic_scales = np.zeros((N_params,)) # note this gets saved as a list (for serialization)
                                                #  todo fix code to unpack it
for i_param in range(N_params):
    characteristic_scales[i_param] = 2*STDS[i_param]

alpha = 0.05 # NOTE alpha gets scaled for each param in Firefly Dynamics function
beta = 5  # >4 yields chaotic firefly dynamics
absorption = 0.6 # somewhere around 0.5 is good according to Yang

annealing_constant = 0.99 # currently only beta is being annealed

############# network config












########## saving

# save the strings
#saveName = title + ' verbose info.json'
#verboseInfoFile = open(saveName,'w')
#json.dump(verbose,verboseInfoFile,indent=2)
#verboseInfoFile.close()

# repackage the config constants into a dictionary
config_dict = {"N_gen":N_gen,"N_bugs":N_bugs,"N_params":N_params,"N_objectives":N_objectives,
               "MEANS":MEANS,"STDS":STDS,"MAXES":MAXES,"MINS":MINS,"characteristic_scales": characteristic_scales.tolist(),
               "alpha":alpha,"beta":beta,"absorption":absorption,"annealing_constant":annealing_constant,
               "verbose":verbose,"PARAMS":PARAMS,"OBJECTIVES":OBJECTIVES,"config_prefix":config_prefix}

# save the result as json formatted data
saveName = config_prefix + ' config.json'
configFile = open(saveName,'w')
json.dump(config_dict,configFile,sort_keys=True,indent=2)
configFile.close()
