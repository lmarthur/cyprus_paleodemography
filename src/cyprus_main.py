# This script performs the main analysis of the cyprus data

# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pyro
import torch
import pyro.contrib.gp as gp
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, JitTrace_ELBO
from pyro.optim import Adam
print('Packages imported')

# import preprocessed proxy data from .txt file
data = np.loadtxt('./preprocessed_data/cyprus_histogram.txt')
time = data[:,0]
num_settlements = data[:,1]
dt = time[1] - time[0]

# get only the first 12000 years of data
time = time[0:(int(12000/dt))]
num_settlements = num_settlements[0:(int(12000/dt))]

# plot data
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(time, num_settlements, 'x')
ax.set_xlabel('Time (years)')
ax.set_ylabel('Number of settlements')
plt.tight_layout()
plt.savefig('./output/plotting/cyprus_histogram.png')
plt.close()

# translate data into torch tensors
time = torch.tensor(time)
num_settlements = torch.tensor(num_settlements)
# sampling_prob = torch.tensor(0.2)

# set prior population model
pop_0 = 1000
pop_prior = np.zeros(time.shape[0])
beta_prior = np.zeros(time.shape[0])
for i in range(time.shape[0]):
    pop_prior[i] = pop_0 * np.exp(i*dt/2500)
    beta_prior[i] = 1.61803 / pop_prior[i]

dt = torch.tensor(dt)

# Define the model and guide for the Bayesian deconvolution with SVI
def model(time, num_settlements):
    loss_rate = pyro.sample('loss_rate', dist.Gamma(2.61803, 160993))
    scaling_factor = pyro.sample('scaling_factor', dist.Gamma(2.61803, 0.0107869))
    sampling_prob = pyro.sample('sampling_prob', dist.Beta(1.2258, 1.6774))
    # now we treat each time bin as a separate sample
    for i in pyro.plate("data", time.shape[0]):
        pop_est = pyro.sample('pop_est_{}'.format(i), dist.Gamma(2.61803, beta_prior[i]))
        proxy_est = sampling_prob * scaling_factor * pop_est * torch.exp(-loss_rate * (time.shape[0] - i)*dt)
        pyro.sample('obs_{}'.format(i), dist.Normal(proxy_est, torch.sqrt(proxy_est)/sampling_prob), obs=num_settlements[i])

def guide(time, num_settlements):
    loss_rate_loc = pyro.param('loss_rate_loc', torch.tensor(2.61803), constraint=dist.constraints.greater_than_eq(1.0))
    loss_rate_scale = pyro.param('loss_rate_scale', torch.tensor(160993), constraint=dist.constraints.positive)

    scaling_factor_loc = pyro.param('scaling_factor_loc', torch.tensor(2.61803), constraint=dist.constraints.greater_than_eq(1.0))
    scaling_factor_scale = pyro.param('scaling_factor_scale', torch.tensor(0.0107869), constraint=dist.constraints.positive)

    sampling_prob_loc = pyro.param('sampling_prob_loc', torch.tensor(1.2258), constraint=dist.constraints.positive)
    sampling_prob_scale = pyro.param('sampling_prob_scale', torch.tensor(1.6774), constraint=dist.constraints.positive)

    loss_rate = pyro.sample('loss_rate', dist.Gamma(loss_rate_loc, loss_rate_scale))
    scaling_factor = pyro.sample('scaling_factor', dist.Gamma(scaling_factor_loc, scaling_factor_scale))
    sampling_prob = pyro.sample('sampling_prob', dist.Beta(sampling_prob_loc, sampling_prob_scale))

    # now we treat each time bin as a separate sample
    for i in pyro.plate("data", time.shape[0]):
        pop_est_loc = pyro.param('pop_est_loc_{}'.format(i), torch.tensor(2.61803), constraint=dist.constraints.greater_than_eq(1.0))
        pop_est_scale = pyro.param('pop_est_scale_{}'.format(i), torch.tensor(beta_prior[i]), constraint=dist.constraints.positive)
        pyro.sample('pop_est_{}'.format(i), dist.Gamma(pop_est_loc, pop_est_scale))

print('Model and guide defined... \n')

pyro.clear_param_store()
optimizer = Adam({"lr": 0.001})
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

num_steps = 25000
lossarray = np.zeros(num_steps)
print("Beginning SVI Deconvolution... \n")
for step in range(num_steps):
    loss = svi.step(time, num_settlements)
    if step % 1000 == 0:
        print("Step {:>5d} loss = {:0.5g}".format(step, loss / time.shape[0]))
    lossarray[step] = loss/time.shape[0]

# plot loss
plt.plot(lossarray)
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Deconvolution Model Loss")
plt.savefig('./output/plotting/cyprus_decon_loss.png')
plt.close()

pop_est_alpha = np.zeros(time.shape[0])
pop_est_beta = np.zeros(time.shape[0])
pop_est_map = np.zeros(time.shape[0])
pop_est_sd = np.zeros(time.shape[0])
pop_est_mean = np.zeros(time.shape[0])
pop_est_lq = np.zeros(time.shape[0])
pop_est_uq = np.zeros(time.shape[0])

for i in range(time.shape[0]):
    pop_est_alpha[i] = pyro.param('pop_est_loc_{}'.format(i)).item()
    pop_est_beta[i] = pyro.param('pop_est_scale_{}'.format(i)).item()
    pop_est_map[i] = (pop_est_alpha[i]-1)/pop_est_beta[i]
    pop_est_mean[i] = pop_est_alpha[i]/pop_est_beta[i]
    pop_est_sd[i] = np.sqrt(pop_est_alpha[i])/pop_est_beta[i]
    pop_est_lq[i] = stats.gamma.ppf(0.25, pop_est_alpha[i], scale=1/pop_est_beta[i])
    pop_est_uq[i] = stats.gamma.ppf(0.75, pop_est_alpha[i], scale=1/pop_est_beta[i])

# get the estimates of the loss rate, scaling factor, and sampling probability
loss_rate_alpha = pyro.param('loss_rate_loc').item()
loss_rate_beta = pyro.param('loss_rate_scale').item()

scaling_factor_alpha = pyro.param('scaling_factor_loc').item()
scaling_factor_beta = pyro.param('scaling_factor_scale').item()

sampling_prob_alpha = pyro.param('sampling_prob_loc').item()
sampling_prob_beta = pyro.param('sampling_prob_scale').item()

# output the estimates of the parameters in a text file
np.savetxt('./output/data/cyprus_decon_params.txt', np.array([loss_rate_alpha, loss_rate_beta, scaling_factor_alpha, scaling_factor_beta, sampling_prob_alpha, sampling_prob_beta]))

# export time, pop_est, pop_est_unc to .txt
np.savetxt('./output/data/cyprus_output.txt', np.transpose([time, pop_est_map, pop_est_mean, pop_est_lq, pop_est_uq]))

print("Results exported \n")