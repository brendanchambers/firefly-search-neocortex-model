import numpy as np
import json
from pprint import pprint



###### firefly config

config_prefix = '1-18-2016' # for making the filestring

verbose = 'compare weight distributions' # optionally provide a general description of the current endeavor
PARAMS = ['p_ei','p_ie','p_ii','w_input'] # ,'lognorm_sigma'] # name for easier printing
OBJECTIVES = ['stable duration','rate_score','asynchrony_score'] # '['asynchrony','stable duration'] # names for easier printing & reference
                    # (for now the second obj dimension is not necessary)

N_gen = 10 # working towards 100+
N_bugs = 30
N_params = len(PARAMS)
N_objectives = len(OBJECTIVES)

# range for [p_ei, p_ie, p_ii, w_input]
MEANS = [0.15, 0.15, 0.2, 10] # ,-1] # for each param
STDS = [0.1, 0.1, 0.1, 2] # ,3]
MAXES = [0.3, 0.3, 0.5, 15] # , 5]
MINS = [0.1, 0.1, 0.05, 5] # , -10]  # sigma must be > 0


characteristic_scales = np.zeros((N_params,)) # note this gets saved as a list (for serialization)
                                                #  todo fix code to unpack it
                                                        #  note I think I took care of this, double check
for i_param in range(N_params):
    characteristic_scales[i_param] = 2*STDS[i_param]

alpha = 0.025 # NOTE alpha gets scaled for each param in Firefly Dynamics function
beta = 4.25  # >4 yields chaotic firefly dynamics
absorption = 0.5 # somewhere around 0.5 is good according to Yang

annealing_constant = 0.999 # currently only beta is being annealed

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
