import json
import numpy as np
import scipy as sp
from brian2 import *
import time


duration = 1 *ms
t_res = 0.1 * ms
input_mean = 5
input_sigma = 1
num_steps = (duration / t_res) + 1
num_cells = 4
cell1_input = np.random.normal(input_mean, input_sigma, (num_steps, num_cells))
print cell1_input
timedArray = TimedArray(cell1_input * mV, dt=t_res)
#print(timedArray(0.3*ms))

G_input = NeuronGroup(num_cells, 'v = timedArray(t,i) : volt')
mon = StateMonitor(G_input, 'v', record=True)
net = Network(G_input, mon)
net.run(duration)

print 'monitor voltage: '
print mon.v

print shape(mon.v)

# todo incorporate this into a sample model
        # by setting a g_input conductance (in excitatory model neurons) using a TimedArray
        # set the latter portion of timebins to zero



