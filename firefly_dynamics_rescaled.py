import numpy as np
import scipy as sp

from memory_profiler import profile

def firefly_dynamics_rescaled(oldPopulation, scores, alpha, beta, absorption, characteristicScales, MAXES, MINS):

    # NOTE this implementation assumes this is a maximization problem

    N_bugs = oldPopulation.shape[1]
    N_params = oldPopulation.shape[0]
    N_objectives = scores.shape[1] # number of scores, i.e. number of objective functions (scores is flies x objectives)

    numUpdatesAdjustment = N_bugs*N_bugs*1.0 / (5*N_objectives) # scaling up to the constant for expected number of updates

    # init
    newPopulation = oldPopulation
    noiseTerms = np.zeros((N_bugs,N_params))
    attractionTerms = np.zeros((N_bugs,N_params))
    dominatedByOthers = np.zeros((N_bugs,1))

    #updateCounts = np.zeros((N_bugs,N_bugs)) # i,j  means j dominated i

    for i_fly in range(0,N_bugs):
        for i_score in range(0,N_objectives):
            if np.isnan(scores[i_fly][i_score]):
                print "warning: unexpected nan in scores"
                scores[i_fly][i_score] = -np.inf  # (pesky nan problems) ultimately we should be able to omit this check

    for i_fly in range(0,N_bugs):
        for j_fly in range(0,N_bugs):

            # NOTE this is a maximization framework

            # if all entries in j are more optimal than i or equal, and if at least one entry is different (i!=j):
            if (scores[i_fly,:] <= scores[j_fly,:]).all() and (scores[i_fly,:] != scores[j_fly,:]).any():

                dominatedByOthers[i_fly] += 1
                #updateCounts[i_fly][j_fly] += 1 # thinking about possible scaling problems
                difference = oldPopulation[:,j_fly] - oldPopulation[:,i_fly]
                differenceSquared = difference**2 # element-wise square
                        # in the Yang implementation, difference squared was L2 distance squared
                        # todo range...is this is fuxd? - big dimensions lead to small attraction terms??
                        # but the real puzzle is why are the attraction terms so BIG for dim 0,1,2 (the small ones)

                alphaRescaling = characteristicScales
                noiseTerm = alpha * sp.randn(N_params) * alphaRescaling # gaussian noise with std dev = alph
                noiseTerms[i_fly,:] += noiseTerm.T # keep track of total noise displacement
                                                                    # todo we could scale this to be invariant to num_flies

                #attractionTerm = beta * np.exp(-absorption * differenceSquared.T) * difference.T  # old version
                #absorptionRescaling = 1.0 / ( (4*characteristicScales) ** 2) # we don't think this is necessary

                #
                absorptionRescaling = 1.0 / (4 * characteristicScales ** 2)  # rescale the absorption exponential
                                        # the constant term 4 is just an arbitrary choice
                attractionTerm = beta * np.exp(-absorption * differenceSquared.T * absorptionRescaling) * difference.T

                print "absorption rescaling " + str(absorptionRescaling)
                print "difference " + str(difference)
                print " attractionTerms " + str(attractionTerm)

                print beta
                print "inner " + str(-absorption * differenceSquared.T)
                print "expd " + str(np.exp(-absorption * differenceSquared.T))
                print "diff " + str(difference.T)
                print "diff sq " + str(differenceSquared.T)
                print attractionTerm

                attractionTerms[i_fly,:] += attractionTerm # keep track of total attraction displacement
                                                                # we could scale this by expected number of steps as a function of population size
        attractionTerms[i_fly,:] = attractionTerms[i_fly,:] * 1.0 / numUpdatesAdjustment # scaling
        noiseTerms[i_fly,:] = noiseTerms[i_fly,:] # * 1.0 / numUpdatesAdjustment # scaling
        newPopulation[:, i_fly] += (attractionTerms[i_fly,:].T + noiseTerms[i_fly,:].T)

    # grab the pareto front
    paretoIDs = [idx for idx, val in enumerate(dominatedByOthers) if val==0]

    # check for all scores == 0
    if (dominatedByOthers==0).all():

        # if all scores are identical, still take a noise step for each fly
        alphaRescaling = characteristicScales
        noiseTerm = alpha * sp.randn(N_bugs,N_params) * alphaRescaling
        newPopulation += noiseTerms.T

        # and respawn some flies randomly
        NUM_SPAWN = sp.round(1.0*N_bugs / 10.0) # respawn 10% of flies
        cullIDs = sp.random.randint(0,N_bugs,NUM_SPAWN)

        # remove cullIDs unless they happen to be part of a pareto optimum for a different score-dimension
        for cullIdx in cullIDs:
            # randomly respawn these bad flies t
            for i_param in range(N_params):
                newPopulation[i_param, cullIdx] = np.random.rand()
                newPopulation[i_param, cullIdx] *= (MAXES[i_param] - MINS[i_param])
                newPopulation[i_param, cullIdx] += MINS[i_param]

    else: # else not all score vectors are identical

        # get the worst flies and respawn them
        # NOTE could pick off bad fireflies at this point...but things seem to be working well enough so far without culling
                # todo do this with max(dominatedByOthers) instead of looping through everything again
        #paretoIDs = []
        cullIDs = []
        for i_score in range(np.shape(scores)[1]):
            scores1D = scores[:, i_score]
            bestScore = np.nanargmax(scores1D)
            worstScore = np.nanargmin(scores1D)  # so if every entry is nan I think this will just choose one arbitrarily
            #paretoIDs.append(bestScore)
            cullIDs.append(worstScore)  # randomly respawn bad flies - NOTE would be better to introduce some stochasticity here
             # handle the case where all scores == 0

        # remove cullIDs unless they happen to be part of a pareto optimum for a different score-dimension
        for cullIdx in cullIDs:
            if ~(cullIdx in paretoIDs):  # if it happens to be part of the pareto front then leave it
                # randomly respawn these bad flies t
                for i_param in range(N_params):
                    newPopulation[i_param, cullIdx] = np.random.rand()
                    newPopulation[i_param, cullIdx] *= (MAXES[i_param] - MINS[i_param])
                    newPopulation[i_param, cullIdx] += MINS[i_param]

    return {'newPopulation':newPopulation,'attractionTerms':attractionTerms,'noiseTerms':noiseTerms,
            'dominatedByOthers':dominatedByOthers,'cullIDs':cullIDs,'paretoIDs':paretoIDs} # ,'updateCounts':updateCounts}


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

