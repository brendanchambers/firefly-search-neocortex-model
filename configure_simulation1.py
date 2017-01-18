from brian2 import *
import json

# todo test this


config_prefix = 'check asyn measure 12-19-2016'  # for making the filestring - this should match the firefly config (todo make this automatic)

######## simulation parameters #############
N_input = 200  # let's add an input population
N_e = 3200
N = 4000
N_i = N - N_e  # inhibitory neurons

duration = "0.3 second"  # total duration (format as a string as shown)
input_duration = "0.3 second"  # inject some current (format as a string as shown)
input_rate = 4 * Hz  # Hz
# input_current = 0 # nA

initial_Vm = "(-60 + 10 * (numpy.random.rand(1, self.N) - 0.5)) * mV"  # this is going to get read in as a string in the networkHelper class

######## network parameters ######
C = 281 * pF
gL = 30 * nS
taum = C / gL
EL = -70.6 * mV
VT = -50.4 * mV
DeltaT = 2 * mV
Vcut = VT + 5 * DeltaT

#w_input = 3 * nS # leave this free (important for regulating state - brunel 2000)
we = 1 * nS  # excitatory synaptic weight (this gets multiplied by a logrand so make sure we get the scaling right)
wi = 10 * nS  # inhibitory synaptic weight

p_connect_input = 0.2  # P(input cell -> excitatory cells)
p_connect_ee = 0.15  # connection probabilities
#p_connect_ei = 0.4
#p_connect_ie = 0.2
#p_connect_ii = 0.3

logrand_sigma = 0.1  # leave these out and examine this continuum computationally
logrand_mu = log(1) - 0.5 * (logrand_sigma ** 2)  # this condition ensures that the mean of the new distributoin = 1

LOG_RAND_sigmaInh = 0.1  # suggesting we hold these constant and only vary excitatory connections
LOG_RAND_muInh = log(1) - 0.5 * (LOG_RAND_sigmaInh ** 2)  # (so we aren't scaling total weight as we explore heavy-tailedness)

# Pick an electrophysiological behaviour
tauw, a, b, Vr, EE, EI, taue, taui = 144 * ms, 4 * nS, 0.0805 * nA, -70.6 * mV, in_unit(0 * mV,mV), -75 * mV, 10 * ms, 5 * ms  # Regular spiking (as in the paper)
# tauw,a,b,Vr=20*ms,4*nS,0.5*nA,VT+5*mV # Bursting
# tauw,a,b,Vr=144*ms,2*C/(144*ms),in_unit(0*nA,nA),-70.6*mV # Fast spiking

########### dynamics of the model ############
# ok to get rid of this one fyi:
eqs = """
dvm/dt = ( -self.gL*(vm-self.EL) + self.gL*self.DeltaT*exp((vm-self.VT)/self.DeltaT) - gE*(vm-self.EE) - g_input*(vm-EE) - gI*(vm-self.EI) - w )/self.C : volt
dgE/dt = -gE/self.taue : siemens
dw/dt = (self.a*(vm - self.EL) - w)/self.tauw : amp
dgI/dt = -gI/self.taui : siemens
g_input = g_input_timedArray(t,i) : siemens
"""

eqs = """
dvm/dt = ( -gL*(vm-EL) + gL*DeltaT*exp((vm-VT)/DeltaT) - gE*(vm-EE) - g_input_timedArray(t,i)*(vm-EE) - gI*(vm-EI) - w )/C : volt
dgE/dt = -gE/taue : siemens
dw/dt = (a*(vm - EL) - w)/tauw : amp
dgI/dt = -gI/taui : siemens
g_input = g_input_timedArray(t,i) : siemens
"""

print eqs



############# write to json file

# using this double-naming idiocy is silly but hopefully very readable (everything is already defined above)
networkconfig_dict = {"N_input":N_input,"N_e":N_e,"N":N,"N_i":N_i,

          "duration":str(duration), "input_duration":str(input_duration),  # make these into strings so they
          "input_rate":str(input_rate),"initial_Vm":str(initial_Vm),           #   can be serialized
          "C":str(C),"gL":str(gL),"taum":str(taum),"EL":str(EL),"VT":str(VT),"DeltaT":str(DeltaT),
          "Vcut":str(Vcut),"we":str(we),"wi":str(wi), # "w_input":str(w_input),  #actively optimizing w_input now

           "p_connect_input":p_connect_input,"p_connect_ee":p_connect_ee,
             #"p_connect_ei":p_connect_ei,p_connect_ie":p_connect_ie,
            #"p_connect_ii":p_connect_ii,
           #"LOG_RAND_sigma":LOG_RAND_sigma,"LOG_RAND_mu":LOG_RAND_mu,
           "LOG_RAND_sigmaInh":LOG_RAND_sigmaInh,"LOG_RAND_muInh":LOG_RAND_muInh,
            "logrand_sigma":logrand_sigma, "logrand_mu":logrand_mu,

           "tauw":str(tauw),"a":str(a),"b":str(b),"Vr":str(Vr),"EE":str(EE),"EI":str(EI),
            "taue":str(taue),"taui":str(taui),

            "eqs":eqs}

# save in json format
saveName = config_prefix + ' networkconfig 1.json'
configFile = open(saveName,'w')
json.dump(networkconfig_dict,configFile,sort_keys=True,indent=2)
configFile.close()


# temp, testing recovery of the physical units outside the config file
print json.dumps(str(duration))
test = str(duration)
print test
print test[0]
print test.split(' ')

testy = test.split(' ')
print testy[0]
print testy[1]

test = "100."
test2 = float(test) # double checking
print test2