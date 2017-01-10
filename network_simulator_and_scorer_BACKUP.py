import json
import numpy as np
from brian2 import *
import matplotlib.pyplot as plt
import time
import scipy.ndimage.filters as filt
import scipy.stats as stats
from brian2.units.allunits import *

# in this version of our quest, only lognorm_sigma and lognorm_mu will be changing
# function of arguments lognorm_sigma and lognorm_mu


#networkconfig_filestring =

class NetworkHelper:

    def __init__(self, networkconfig_filestring):

        # load the networkconfig file
        networkconfig_file = open(networkconfig_filestring, 'r')
        with networkconfig_file as data_file:
            network_config = json.load(data_file)
        networkconfig_file.close()

        print('network config file from ' + networkconfig_filestring)

        self.N_input = network_config['N_input']
        self.N_e = network_config['N_e']
        self.N_i = network_config['N_i']
        self.N = network_config['N']

        parsed = network_config['duration'].split(' ') # read in string representation
        print str(parsed[0]) + ' * ' + str(parsed[1])
        self.duration = eval(str(parsed[0]) + ' * ' + str(parsed[1])) # convert to physical units
        parsed = network_config['input_duration'].split(' ')
        self.input_duration = eval(parsed[0] + ' * ' + parsed[1])
        parsed = network_config['input_rate'].split(' ')
        self.input_rate = eval(parsed[0] + ' * ' + parsed[1])

        parsed = network_config['initial_Vm']
        print 'initial voltage distribution rule: ' + parsed
        self.initial_Vm = eval(parsed)

        parsed = network_config['C'].split(' ')
        self.C = eval(parsed[0] + ' * ' + parsed[1])
        parsed = network_config['gL'].split(' ')
        self.gL = eval(parsed[0] + ' * ' + parsed[1])
        parsed = network_config['taum'].split(' ')
        self.taum = eval(parsed[0] + ' * ' + parsed[1])
        parsed = network_config['EL'].split(' ')
        self.EL = eval(parsed[0] + ' * ' + parsed[1])
        parsed = network_config['VT'].split(' ')
        self.VT = eval(parsed[0] + ' * ' + parsed[1])
        parsed = network_config['DeltaT'].split(' ')
        self.DeltaT = eval(parsed[0] + ' * ' + parsed[1])
        parsed = network_config['Vcut'].split(' ')
        self.Vcut = eval(parsed[0] + ' * ' + parsed[1])

        parsed = network_config['w_input'].split(' ')
        self.w_input = eval(parsed[0] + ' * ' + parsed[1])
        parsed = network_config['we'].split(' ')
        self.we = eval(parsed[0] + ' * ' + parsed[1])
        parsed = network_config['wi'].split(' ')
        self.wi = eval(parsed[0] + ' * ' + parsed[1])

        # todo the non-physical units
        self.p_connect_input = network_config['p_connect_input']
        self.p_connect_ee = network_config['p_connect_ee']
        #self.p_connect_ie = network_config['p_connect_ie']
        #self.p_connect_ei = network_config['p_connect_ei']
        self.p_connect_ii = network_config['p_connect_ii']

        self.logrand_sigma = network_config['logrand_sigma']
        self.logrand_mu = network_config['logrand_mu']
        self.LOG_RAND_sigmaInh =  network_config['LOG_RAND_sigmaInh']
        self.LOG_RAND_muInh = network_config['LOG_RAND_muInh']

        parsed = network_config['tauw'].split(' ')
        self.tauw = eval(parsed[0] + ' * ' + parsed[1])
        parsed = network_config['a'].split(' ')
        self.a = eval(parsed[0] + ' * ' + parsed[1])
        parsed = network_config['b'].split(' ')
        self.b = eval(parsed[0] + ' * ' + parsed[1])
        parsed = network_config['Vr'].split(' ')
        self.Vr = eval(parsed[0] + ' * ' + parsed[1])
        parsed = network_config['EE'].split(' ')
        self.EE = eval(parsed[0] + ' * ' + parsed[1])
        parsed = network_config['EI'].split(' ')
        self.EI = eval(parsed[0] + ' * ' + parsed[1])
        parsed = network_config['taue'].split(' ')
        self.taue = eval(parsed[0] + ' * ' + parsed[1])
        parsed = network_config['taui'].split(' ')
        self.taui = eval(parsed[0] + ' * ' + parsed[1])

        self.eqs = Equations(str(network_config['eqs'])) # dynamics of the model


    def  simulateActivity(self, input_args, verboseplot=False):
        p_ei = input_args[0]
        p_ie = input_args[1]
        ########### define the neurons and connections
        logrand_mu = log(1) - 0.5*(self.logrand_sigma**2) # this establishes mean(W) = 1, regardless of sigma

        # having trouble with the self stuff in passing string arguments to Brian2
        #  so just pursue the course of least resistance - rename the variables todo find the real solution

        initial_Vm = self.initial_Vm

        C = self.C
        gL = self.gL
        taum = self.taum
        EL = self.EL
        VT = self.VT
        DeltaT = self.DeltaT
        Vcut = self.Vcut

        tauw = self.tauw
        a = self.a
        b = self.b
        Vr = self.Vr
        EE = self.EE
        EI = self.EI
        taue = self.taue
        taui = self.taui

        w_input = self.w_input
        we = self.we
        wi = self.wi

        eqs = self.eqs  # dynamics of the model

        '''
        print eqs.diff_eq_names
        print type(eqs)
        print Vr
        print type(Vr)
        print b
        print type(b)
        '''
        #input_units = PoissonGroup(self.N_input, self.input_rate)
        input_units = PoissonGroup(self.N_input, self.input_rate)
        neurons = NeuronGroup(self.N, model=eqs, threshold='vm>Vcut',
                              reset="vm=Vr; w+=b", refractory=1*ms, method='euler')

        Pe = neurons[:self.N_e]  # excitatory subpopulation
        Pi = neurons[self.N_e:self.N]

        Cinput = Synapses(input_units, Pe, on_pre='gE+=w_input')
        Cee = Synapses(Pe, Pe, model='''alpha : 1''', on_pre='gE+=(alpha*we)')  # define the synapse groups
        Cei = Synapses(Pe, Pi, model='''alpha : 1''', on_pre='gE+=(alpha*we)')
        Cii = Synapses(Pi, Pi, model='''alpha : 1''', on_pre='gI+=(alpha*wi)')
        Cie = Synapses(Pi, Pe, model='''alpha : 1''', on_pre='gI+=(alpha*wi)')

        # note - we can also specify vectors Pre and Post, so that Pre(i) -> Post(i) ... e.g. Cee.connect(i=Pre, j=Post)
        Cinput.connect(p=self.p_connect_input)
        Cee.connect(p=self.p_connect_ee)  # p_connect_ee) # connect randomly  (additional argument we might want: condition='i!=j')
        #Cei.connect(p=self.p_connect_ei)
        Cii.connect(p=self.p_connect_ii)
        #Cie.connect(p=self.p_connect_ie)
        Cei.connect(p=p_ei)
        Cie.connect(p=p_ie) # passed in as argument

        N_ee = len(Cee)  # todo we don't want to do this (but I'm getting errors because e.g. len(alpha)!=len(Cee)
        N_ie = len(Cie)
        N_ei = len(Cei)
        N_ii = len(Cii)

        Cii.alpha = numpy.random.lognormal(self.LOG_RAND_muInh, self.LOG_RAND_sigmaInh, N_ii)  # NOTE mean of these distributions = 1
        Cie.alpha = numpy.random.lognormal(self.LOG_RAND_muInh, self.LOG_RAND_sigmaInh, N_ie)  # TODO remove the autapses on the main diagonal
        Cei.alpha = numpy.random.lognormal(self.logrand_mu, self.logrand_sigma, N_ei)
        Cee.alpha = numpy.random.lognormal(self.logrand_mu, self.logrand_sigma, N_ee)
        '''
        # this shows how to approach this without knowing e.g. N_ei ahead of time
        Cii.alpha = numpy.random.lognormal(LOG_RAND_mu, LOG_RAND_sigma, N_i * N_i * p_connect_ii) # define alpha
        Cie.alpha = numpy.random.lognormal(LOG_RAND_mu, LOG_RAND_sigma, N_i * N_e * p_connect_ie)  #  TODO remove the autapses on the main diagonal
        Cei.alpha = numpy.random.lognormal(LOG_RAND_mu, LOG_RAND_sigma, N_e * N_i * p_connect_ei)  # ERROR b/c allpha has different size each iteration
        Cee.alpha = numpy.random.lognormal(LOG_RAND_mu, LOG_RAND_sigma, N_e * N_e * p_connect_ee)
        '''

        ################# run the simulation ####################
        ###### initialization
        neurons.vm = self.initial_Vm
        neurons.gE = 0 * nS
        neurons.gI = 0 * nS
        # neurons.I = input_current * nA # current injection during input period

        ###### recording activity
        s_mon = SpikeMonitor(neurons)  # keep track of population firing
        P_patch = neurons[(self.N_e - 1):(self.N_e + 1)]  # random exc and inh cell #todo chose these randomly
        dt_patch = 0.1 * ms  # 1/sampling frequency  in ms
        patch_monitor = StateMonitor(P_patch, variables=('vm', 'gE', 'gI', 'w'), record=True,
                                     dt=dt_patch)  # keep track of a few cells
        inputRate_monitor = PopulationRateMonitor(input_units)

        ###### stimulation period
        #run(self.input_duration)
        run(self.input_duration)

        ######  non-stimulation period  % todo define some input connectivity
        # neurons.I = 0 * nA # current injection turned off
        # input_units = PoissonGroup(N_input, 0*Hz) # turn off input firing
        # Cinput = Synapses(input_units, Pe, on_pre = 'gE+=w_input')
        # Cinput.connect(p=p_connect_input)

        # input_units = PoissonGroup(N_input, 0*Hz) # silence the inputs - this doesn't work
        # (# note - looks like the TimedArray Brian2 class has potential to be a better solution for dynamic inputs)
        run(self.duration - self.input_duration)  # make everything go

        # device.build(directory='output', compile=True, run=True, debug=False) # for the c++ build (omitting during testing)
        # todo try to get this back in if we want it to run faster








        ################ plotting    todo don't actually need this here    ####################
        if verboseplot:
            ##### spiking activity
            plt.figure
            plt.plot(s_mon.t / ms, s_mon.i, '.k')
            plt.xlabel('Time (ms)')
            plt.ylabel('Neuron index')
            plt.title('population spiking')
            plt.show()

            ####  example excitatory neuron
            tt = patch_monitor.t / ms
            f, axarr = plt.subplots(5, sharex=True)
            plt.title('an excitatory neuron')
            axarr[0].plot(tt, patch_monitor[0].vm, 'k')
            axarr[0].set_ylabel('Vm')
            axarr[1].plot(tt, patch_monitor[0].gE, 'b')
            axarr[1].set_ylabel('gE')
            axarr[2].plot(tt, patch_monitor[0].gI, 'r')
            axarr[2].set_ylabel('gI')
            axarr[3].plot(tt, patch_monitor[0].w, 'ro')
            axarr[3].set_ylabel('adaptation w')
            axarr[4].plot(tt, inputRate_monitor.smooth_rate(width=5 * ms) / Hz, 'k')
            axarr[4].set_ylabel('input rate')
            plt.xlabel('time (ms)')
            plt.show()

            #####  example inhibitory neuron
            tt = patch_monitor.t / ms
            f, axarr = plt.subplots(5, sharex=True)
            plt.title('an inhibitory neuron')
            axarr[0].plot(tt, patch_monitor[1].vm, 'k')
            axarr[0].set_ylabel('Vm')
            axarr[1].plot(tt, patch_monitor[1].gE, 'b')
            axarr[1].set_ylabel('gE')
            axarr[2].plot(tt, patch_monitor[1].gI, 'r')
            axarr[2].set_ylabel('gI')
            axarr[3].plot(tt, patch_monitor[1].w, 'ro')
            axarr[3].set_ylabel('adaptation w')
            axarr[4].plot(tt, inputRate_monitor.smooth_rate(width=5 * ms) / Hz, 'k')
            axarr[4].set_ylabel('input rate')
            plt.xlabel('time (ms)')
            plt.show()







###################################### scoring

        # reformat the spike times

        # todo smooth? subsample? (subsampling might be cheaper but smoothing so that bin size doesn't affect the score would be nice)

        # BIN SIZE
        binwidth = 1 # ms
        smoothSigma = 4 # bins
        numBins = np.ceil((self.duration/ms) / binwidth)+1 #  https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve1d.html
        #meanSmoothRate
        raster = np.zeros((self.N,numBins))

        for i_spike in range(0,len(s_mon.i)):
            cellIdx = s_mon.i[i_spike]
            binIdx = np.round((s_mon.t[i_spike] / ms) / binwidth)
            raster[cellIdx][binIdx] += 1   # this throws away a lot of information # todo fix this

        raster = filt.gaussian_filter1d(raster,sigma=smoothSigma,axis=1) # todo should smooth BEFORE rebinning

        # compute corr coeffs of rates  # todo do this better
        corrcoeffs = np.corrcoef(raster[0:self.N_e][:])  # look at excitatory cells only
                    # todo mask out the diagonal (replace with nans)
        np.fill_diagonal(corrcoeffs, np.nan)
        print "sample diagonal element - " , corrcoeffs[5][5] # testing

        if verboseplot:
            # check the processed raster
            plt.figure
            plt.imshow(raster,interpolation='nearest',aspect='auto')
            plt.tight_layout()
            plt.gray()
            plt.title('how does the raster look after smoothing')
            plt.show()

            print 'shape of the corroeffs ' , shape(corrcoeffs)
            print " mean corr coeff " + str(mean(corrcoeffs[~np.isnan(corrcoeffs)]))
            print " mean abs corr coeff " + str(mean(np.abs(corrcoeffs[~np.isnan(corrcoeffs)])))

            plt.figure()
            plt.hist(corrcoeffs[~np.isnan(corrcoeffs)],bins='auto')
            plt.show()
            plt.title('distributon of correlation coefficients')

 ######################    scaling with K      take multiple samples from the corrcoeffs # todo this, didn't really work
        sample_sizes = [100, 300, 500, 750, 1000]
        num_repeats = 10
        samples = np.zeros((num_repeats,len(sample_sizes)))

        for i_sample in arange(len(sample_sizes)):
            K = sample_sizes[i_sample]
            for i in range(num_repeats):
                sampleIdxs = np.random.choice(self.N_e,K)
                sampleCorrCoeffs = corrcoeffs[np.ix_(sampleIdxs,sampleIdxs)]

                # todo put the means and medians into a numpy array and plot them
                samples[i][i_sample] = np.nanmedian(sampleCorrCoeffs)

        slope, intercept, r_val, p_val, std_err = stats.linregress(sample_sizes, np.mean(samples,0))  # todo could do this better (don't avg beforehand)
        if verboseplot:
            # plot the range of samples
            plt.figure()
            for i_repeat in np.arange(num_repeats):
                plt.plot(sample_sizes, samples[i_repeat][:].T) # todo remove connector line
            plt.plot(sample_sizes,np.mean(samples,0))
            plt.title('sample size vs median corr coef')
            plt.show()

#########################     find the landmarks



###########################      package it all up


        #asynchrony_score = 1 - mean(np.abs(corrcoeffs[~np.isnan(corrcoeffs)])) # this isn't working well
        asynchrony_score =  -slope - mean(corrcoeffs[~np.isnan(corrcoeffs)]) # (reward negative slope and low mean corrcoef)
        #asynchrony_score = -(np.sum(np.abs(corrcoeffs[~np.isnan(corrcoeffs)])))  # minimize the corr corrcoeffs (= maximize -1 * sumsquare corrcoeffs)

        NUM_COMPONENTS = 2 # temp - asynchrony and null # todo read this in automatically
        score_components = np.zeros((NUM_COMPONENTS,))
        score_components[0] = np.random.randn() # asynchrony_score # it's named like this because the plan used to be, pareto front
        score_components[1] = np.random.randn() # temp
        return score_components



## main:
'''
test = NetworkHelper('firefly cleanup 12-6-2016 networkconfig.json')

logrand_sigma_test = 1
logrand_mu_test = log(1) - 0.5 * (logrand_sigma_test ** 2)
test.simulateActivity(logrand_sigma_test, logrand_mu_test)
'''

'''
OLD MATLAB CODE FOR REFERENCE
GAUSS_WIDTH = 3; % ms
raster_smooth = zeros(size(raster));
filter = gausswin(GAUSS_WIDTH ./ TICK); % translate from ms to number of points
for i=1:size(raster,1)
raster_smooth(i,:) = conv(raster(i,:),filter,'same');
end
figure(); imagesc(raster_smooth); colorbar; title('smoothed spike raster');

%% obj function components
% asynchronous (pairwise correlation coefficients centered near 0)
%CCs = zeros(1,(N+1)*(N/2));
CCs = corrcoef(raster_smooth'); % tranpose to put variables in the columns instead of rows
BINS = 25;
figure(); hist(CCs(:), BINS); title('are corr coefs near 0 ?'); shading flat;

% irregular (sigma over mu > 1 for ISIs) (coefficient of variation)
ISIs = cell(N,1);
CoVs = zeros(1,N);
for i_cell = 1:N
ISIs{i_cell} = diff(spiketimes{i_cell});
CoVs(i_cell) = std(ISIs{i_cell}) / mean(ISIs{i_cell}); % coeff of variation
end
figure(); hist(CoVs,BINS); title('is CoV > 1 ?'); shading flat;

% TODO criticality

% consider the set of active cells:
sumActivity = zeros(1,N);
for i_cell = 1:N
sumActivity(i_cell) = length(spiketimes{i_cell});
end
ACTIVITY_CUTOFF = 0
activeIdxs = find(sumActivity > ACTIVITY_CUTOFF); % NOTE: could speed this up at the cost of readability using logical indexing of 'rates' instead

% firing rate (1 Hz E, 5 Hz I?)
rates = sumActivity ./ (DURATION/1000); % Hz
targetRate_E = 5; % let's just use a 5 Hz excitatory target for now
figure(); hist(rates(activeIdxs), BINS); title('firing rates in the active population');  % NOTE: could score based on fraction of population active if necessary


% high conductance state in active neurons
% mean (~-60) and variance (2 - 6 mV) of subthreshold potential TODO
% can we get a better description of this somewhere?
meanVm = mean(subthresholdSamples,2);
stdVm = std(subthresholdSamples);

'''


