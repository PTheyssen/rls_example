import numpy as np
import matplotlib.pylab as plt
import math
import rls

#-----------------------------------------------------------
# Example: sinus signal
#-----------------------------------------------------------

samples = 300
time = np.linspace(0.1, 5, samples)
true_signal = [math.sin(x) for x in time]
noise = np.random.normal(0, 0.1, samples) 
y = true_signal + noise

rec_ls = rls.rls(samples, 2, time, y, 0.9, 100)

# use estimated parameters to calculate output at each time step
result = []
for i in range(samples):
    par = rec_ls[i]
    t = time[i]
    result.append(par[0] + par[1] * t)



plt.plot(time, true_signal, linestyle='--', color='r',\
label='true signal')
# plt.plot(time, y, 'ro', label='samples')
plt.plot(time, result, color='g', label='recursive ls')
plt.legend()
plt.show()

