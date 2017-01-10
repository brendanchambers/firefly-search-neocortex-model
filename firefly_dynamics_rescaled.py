import numpy as np
import scipy as sp

from memory_profiler import profile

def firefly_dynamics_rescaled(oldPopulation, scores, alpha, beta, absorption, characteristicScales):

    # NOTE this implementation assumes this is a maximization problem

    N_bugs = oldPopulation.shape[1]
    N_params = oldPopulation.shape[0]
    N_objectives = scores.shape[0] # number of scores, i.e. number of objective functions

    # init
    newPopulation = oldPopulation
    noiseTerms = np.zeros((N_bugs,N_params))
    attractionTerms = np.zeros((N_bugs,N_params))
    dominatedByOthers = np.zeros((N_bugs,1))

    for i_fly in range(0,N_bugs):
        for j_fly in range(0,N_bugs):

            # NOTE this is a maximization framework
            # if all entries in j are more optimal than i or equal, and if at least one entry is different (i!=j):
            scores[np.isnan(scores)] = -np.inf # first remove nans and penalize them
            if (scores[i_fly,:] <= scores[j_fly,:]).all() and (scores[i_fly,:] != scores[j_fly,:]).any():

                dominatedByOthers[i_fly] += 1
                difference = oldPopulation[:,j_fly] - oldPopulation[:,i_fly]
                differenceSquared = difference**2 # element-wise square
                        # todo in the Yang implementation, difference squared should be L2 distance squared

                alphaRescaling = characteristicScales
                noiseTerm = alpha * sp.randn(N_params) * alphaRescaling # gaussian noise with std dev = alph
                noiseTerms[i_fly,:] += noiseTerm.T # keep track of total noise displacement
                                                                    # todo we could scale this to be invariant to num_flies
                #absorptionRescaling = 1 / ((characteristicScales*0.1) ** 2) # we don't think this is necessary
                attractionTerm = beta * np.exp(-absorption * differenceSquared.T) * difference.T
                attractionTerms[i_fly,:] += attractionTerm # keep track of total attraction displacement
                                                                # we could scale this by expected number of steps as a function of population size

        newPopulation[:, i_fly] += (attractionTerms[i_fly,:].T + noiseTerms[i_fly,:].T)

    # todo maybe explicitly return the pareto front using dominatedByOthers==0

    return {'newPopulation':newPopulation,'attractionTerms':attractionTerms,'noiseTerms':noiseTerms}


######## test for the function ###########
'''
N_bugs = 5
PARAMS = ['x','y']
N_params = len(PARAMS)
OBJS = ['z','a']
population = np.zeros((N_params, N_bugs))
#print population
scoreVectors = np.zeros((N_bugs, len(OBJS)))

alpha = 0.5
beta = 0.5
absorption = 0.8
firefly_dynamics(population, scoreVectors, alpha, beta, absorption)


### test
#print ([2, 2, 2] <= [2, 3, 2]).all() and ([1, 2, 3] != [1, 2, 3]).any()
'''

