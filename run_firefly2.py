import json
import numpy as np
import scipy as sp
#from benchmark_objs import rosenbrock_obj
from firefly_dynamics_rescaled import firefly_dynamics_rescaled
from network_simulator_and_scorer import NetworkHelper
import time
from memory_profiler import profile

#@profile
def run_firefly2():
    config_filestring = '1-18-2016 config.json'
    networkconfig_filestring = '1-18-2016 networkconfig 2.json'


    ############### load the firefly config file
    config_file = open(config_filestring,'r')
    with config_file as data_file:
        firefly_config = json.load(data_file)
    config_file.close()

    print('CONFIG FILE: ' + firefly_config['verbose'])
    print('from ' + config_filestring)

    save_prefix = firefly_config['config_prefix']

    N_gen = firefly_config['N_gen']  # this is kind of dumb I realize, but hopefully worth it for readability
    N_bugs = firefly_config['N_bugs']
    N_params = firefly_config['N_params']
    N_objectives = firefly_config['N_objectives']

    # range for rosenbrock
    MEANS = firefly_config['MEANS']
    STDS = firefly_config['STDS']
    MAXES = firefly_config['MAXES']
    MINS = firefly_config['MINS']

    characteristic_scales = np.array(firefly_config['characteristic_scales']) # note this gets saved as a list (for serialization)

    alpha = firefly_config['alpha'] # NOTE alpha gets scaled by char scale for each param in Firefly Dynamics function
    beta = firefly_config['beta']  # >4 yields chaotic firefly dynamics
    absorption = firefly_config['absorption'] # somewhere around 0.5 is good according to Yang

    annealing_constant = firefly_config['annealing_constant'] # currently only beta is being annealed


    ############ initializations
    population = sp.randn(N_params, N_bugs)
    for i_bug in range(N_bugs):
        population[:,i_bug] *= STDS   # gaussian scatter around the means using the stds
        population[:,i_bug] += MEANS

    # check bounds on parameter values
    for i_param in range(N_params):
        for i_fly in range(N_bugs):
            if population[i_param,i_fly] < MINS[i_param]:
                population[i_param,i_fly] = MINS[i_param]
            if population[i_param,i_fly] > MAXES[i_param]:
                population[i_param,i_fly] = MAXES[i_param]
    print 'initial population: ', population

    scoreVectors = np.zeros((N_bugs, N_objectives))
    attractionTerms = np.zeros((N_bugs, N_params))
    noiseTerms = np.zeros((N_bugs, N_params))

    oneGen = [dict() for i_bug in range(N_bugs)] # write this dictionary to the json results file after each generation
    f_name = save_prefix + ' results 2.json'
    resultsFile = open(f_name, 'w')
    resultsFile.write("[") # for formatting a valid json object

    network_helper = NetworkHelper(networkconfig_filestring)  # this object does all the simulation work
    if network_helper.__class__.cell_inputs == None: # if not yet initialized
        network_helper.initializeInputs()  # initialize inputs as a global static variable - currently this still gets copied when it gets made into a timed array

    startTime = time.time()

    ################ run the firefly algorithm
    for i_gen in range(0, N_gen):

        print 'generation: ' , i_gen

        # todo  handle meta-heuristics
        beta *= annealing_constant
        for i_fly in range(0, N_bugs): # todo better to enumerate the firebugs directly
            scoreVectors[i_fly,:] = network_helper.simulateActivity(population[:,i_fly],verboseplot=False)
            #scoreVectors[i_fly, 1] = rosenbrock_obj(population[:, i_fly]) # temp just use the same obj for both

        # NOTE could pick off bad fireflies at this point...but things seem to be working well enough so far without culling

        # keep track of progress for plotting etc
        for i_fly in range(0,N_bugs):
            oneGen[i_fly] = {'noise':np.copy(noiseTerms[i_fly,:]).tolist(),'attraction':np.copy(attractionTerms[i_fly,:]).tolist(),
                                            'alpha':alpha,'beta':beta,'absorption':absorption,
                                            'score':np.copy(scoreVectors[i_fly,:]).tolist(),'params':np.copy(population[:,i_fly]).tolist(),
                                            'gen':i_gen,'fly':i_fly}
                                            # NOTE could maybe improve efficiency here...do we need to copy?

        # scale alpha and ?absorption?
        result = firefly_dynamics_rescaled(population, scoreVectors, alpha, beta, absorption, characteristic_scales)
        newPopulation = result['newPopulation']
        attractionTerms = result['attractionTerms']
        noiseTerms = result['noiseTerms']
        population = newPopulation

        # check bounds on parameter values
        for i_param in range(N_params):
            for i_fly in range(N_bugs):
                if population[i_param,i_fly] < MINS[i_param]:
                    population[i_param,i_fly] = MINS[i_param]
                if population[i_param,i_fly] > MAXES[i_param]:
                    population[i_param,i_fly] = MAXES[i_param]

        json.dump(oneGen, resultsFile, sort_keys=True, indent=2)
        if i_gen < (N_gen - 1):
            resultsFile.write(",")

        # todo keep track of the pareto front

    ############### save the results for plotting and post-hoc analysis

    print 'elapsed time for firefly alg: ', time.time() - startTime, ' seconds'

    resultsFile.write("]")
    resultsFile.close()

run_firefly2()