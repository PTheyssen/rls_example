import numpy as np
import matplotlib.pylab as plt
import rls

#-----------------------------------------------------------
# Example: linear signal
#-----------------------------------------------------------

samples = 100
time = np.linspace(0.1, 1, samples)
true_signal = 0.6 + 0.3 * time
noise = np.random.normal(0, 0.1, samples) 
y = true_signal + noise


bls_r = rls.batch_ls(samples, 2, time, y, ridge_factor=1e-05)
rec_ls = rls.rls(samples, 2, time, y, 0.99, 100)

rec_par_0 = rec_ls[samples-1][0]
rec_par_1 = rec_ls[samples-1][1]

par_0 = [x[0] for x in rec_ls]
par_1 = [x[1] for x in rec_ls]



plt.subplot(2, 1, 1)
plt.plot(time, par_0, color='b', label='estimated par_0')
plt.plot(time, par_1, color='r', label='estimated par_1')
plt.plot(time, [0.6]*samples, color='b', linestyle='--', label='true par_0')
plt.plot(time, [0.3]*samples, color='r', linestyle='--', label='true par_1')
plt.legend()

plt.subplot(2, 1, 2)
# plt.plot(time, y, 'ro', label='samples')
plt.plot(time, true_signal, linestyle='--', color='r',\
label='true signal')
plt.plot(time, bls_r[0] + bls_r[1] * time, linestyle='--',\
color='b', label='bls_ridge')
plt.plot(time, rec_par_0 + rec_par_1 * time, color='g',\
linestyle='--', label='recursive ls')
plt.legend()

plt.show()
