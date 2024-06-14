# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pyro
import torch
import pyro.distributions as dist

# set matplotlib style
plt.style.use('ggplot')

# import fully processed proxy data from .txt file
data = np.loadtxt('./output/data/cyprus_output.txt')
textsize = 14
params = {
   'axes.labelsize': 8,
   'font.size': 8,
   'font.family': 'serif',
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': True,
   }
plt.rcParams.update(params)

# plot the MAP and mean estimates with the IQR shaded
fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
plt.style.use('ggplot')
color = 'black'

ax.fill_between(data[:,0], data[:,3], data[:,4], alpha=0.5, label='Interquartile Range')
ax.plot(data[:,0], data[:,2], 'x', label='Mean estimate')
ax.plot(data[:,0], data[:,1], 'x', label='MAP estimate')

ax.axvline(x=-11000, color='black', linestyle='--', linewidth=1)
ax.axvline(x=-9000, color='black', linestyle='--', linewidth=1)
ax.axvline(x=-6800, color='black', linestyle='--', linewidth=1)
ax.axvline(x=-5200, color='black', linestyle='--', linewidth=1)
ax.axvline(x=-4000, color='black', linestyle='--', linewidth=1)
ax.axvline(x=-2400, color='black', linestyle='--', linewidth=1)
ax.axvline(x=-1050, color='black', linestyle='--', linewidth=1)
ax.axvline(x=-475, color='black', linestyle='--', linewidth=1)
ax.axvline(x=58, color='black', linestyle='--', linewidth=1)
ax.axvline(x=400, color='black', linestyle='--', linewidth=1)

ax.annotate('Late \n Epipaleolithic', xy=(-10000, 5000), fontsize=8, fontfamily='serif', ha='center')
ax.annotate('Early \n Aceramic \n Neolithic', xy=(-7900, 8000), fontsize=8, fontfamily='serif', ha='center')
ax.annotate('Late \n Aceramic \n Neolithic', xy=(-6000, 18000), fontsize=8, fontfamily='serif', ha='center')
ax.annotate('Ceramic \n Neolithic', xy=(-4600, 20000), fontsize=8, fontfamily='serif', ha='center')
ax.annotate('Chalcolithic', xy=(-3200, 18000), fontsize=8, fontfamily='serif', ha='center')
ax.annotate('Bronze Age', xy=(-1725, 22000), fontsize=8, fontfamily='serif', ha='center')
ax.annotate('Iron \n Age', xy=(-762.5, 25000), fontsize=8, fontfamily='serif', ha='center')
ax.annotate('Classical \n and \n Hellenistic', xy=(-200, 7500), xytext=(-1725, 7500), fontsize=8, fontfamily='serif', ha='center', arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='black'))
ax.annotate('Roman', xy=(229, 12000), xytext=(800, 15000), fontsize=8, fontfamily='serif', ha='center', arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='black'))
ax.annotate('Medieval \n and \n Byzantine', xy=(1000, 24000), fontsize=8, fontfamily='serif', ha='center', color='black')

ax.set_xlabel('Time (years)', fontsize=textsize, fontfamily='serif')
ax.set_ylabel('Population', fontsize=textsize, fontfamily='serif')
ax.set_title('Cyprus Population Estimate', fontsize=14, fontfamily='serif')
ax.legend(loc='upper left', fontsize=textsize)
plt.savefig('./output/plotting/cyprus_gp_regression.pdf')
plt.tight_layout()
plt.close()

# plot the model parameter posteriors using pyro
parameters = np.loadtxt('./output/data/cyprus_decon_params.txt')

loss_rate_alpha = parameters[0]
loss_rate_beta = parameters[1]

scaling_factor_alpha = parameters[2]
scaling_factor_beta = parameters[3]

sampling_prob_alpha = parameters[4]
sampling_prob_beta = parameters[5]

ax, fig = plt.subplots(1, 3, figsize=(8, 4.5))

x = np.linspace(0, 0.0015, 1000)
fig[0].plot(x, dist.Gamma(loss_rate_alpha, loss_rate_beta).log_prob(torch.tensor(x)).exp().numpy())
fig[0].set_xlabel('Loss Rate', fontsize=textsize, fontfamily='serif')
fig[0].set_ylabel('Probability Density', fontsize=textsize, fontfamily='serif')

fig[0].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
print("Loss Rate Mean: ", loss_rate_alpha / loss_rate_beta,  "Loss Rate Mode: ", (loss_rate_alpha - 1) / loss_rate_beta, "Loss Rate Std: ", np.sqrt(loss_rate_alpha / loss_rate_beta**2))

x = np.linspace(0, 250, 1000)
fig[1].plot(x, dist.Gamma(scaling_factor_alpha, scaling_factor_beta).log_prob(torch.tensor(x)).exp().numpy())
fig[1].set_xlabel('Scaling Factor', fontsize=textsize, fontfamily='serif')
fig[1].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
fig[1].set_title('Model Parameter Distributions', fontsize=16, fontfamily='serif')
print("Scaling Factor Mean: ", scaling_factor_alpha / scaling_factor_beta,  "Scaling Factor Mode: ", (scaling_factor_alpha - 1) / scaling_factor_beta, "Scaling Factor Std: ", np.sqrt(scaling_factor_alpha / scaling_factor_beta**2))

x = np.linspace(0, 0.25, 1000)
fig[2].plot(x, dist.Beta(sampling_prob_alpha, sampling_prob_beta).log_prob(torch.tensor(x)).exp().numpy())
fig[2].set_xlabel('Sampling Probability', fontsize=textsize, fontfamily='serif')
fig[2].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
print("Sampling Probability Mean: ", sampling_prob_alpha / sampling_prob_beta,  "Sampling Probability Mode: ", (sampling_prob_alpha - 1) / sampling_prob_beta, "Sampling Probability Std: ", np.sqrt(sampling_prob_alpha / sampling_prob_beta**2))

plt.tight_layout()
plt.savefig('./output/plotting/cyprus_parameter_distributions.pdf')