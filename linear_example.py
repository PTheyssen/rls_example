import numpy as np
import matplotlib.pylab as plt
import rls

#-----------------------------------------------------------
# Example: linear signal
#-----------------------------------------------------------

samples = 200
time = np.linspace(0.1, 1, samples)
true_signal = 0.6 + 0.3 * time
noise = np.random.normal(0, 0.1, samples) 
y = true_signal + noise


bls_r = rls.batch_ls(samples, 2, time, y, ridge_factor=1e-05)
rec_ls = rls.rls(samples, 2, time, y, 0.99, 100)

# use estimated parameters to calculate output at each time step
result = []
for i in range(samples):
    par = rec_ls[i]
    t = time[i]
    result.append(par[0] + par[1] * t)



plt.plot(time, y, 'ro', label='samples')
plt.plot(time, true_signal, linestyle='--', color='r',\
label='true signal')
plt.plot(time, bls_r[0] + bls_r[1] * time, linestyle='--',\
color='b', label='bls_ridge')
plt.plot(time, result, color='g', label='recursive ls')
plt.legend()
plt.show()
