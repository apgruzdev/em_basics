# # Imports

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scipy import stats

import matplotlib.pyplot as plt
# %matplotlib inline

# # Preface

np.random.seed(seed=42)


# # EM

# ## Generate data

@dataclass
class GaussParam:
    """Storage for normal distribution parameters"""
    mu: float
    sigma: str


# +
dists_pars = [GaussParam(mu=5., sigma=3.), GaussParam(mu=15., sigma=4.)]
samples_nums = [500, 700]

assert len(dists_pars) == len(samples_nums) == 2, "shapes must be equal (currently 2)"
# -

samples = [np.random.normal(loc=dists_par.mu, scale=dists_par.sigma, size=smpl_num) for dists_par, smpl_num in zip(dists_pars, samples_nums)]
samples_stack = np.concatenate(samples)

for sample in samples:
    plt.hist(sample, bins=50, alpha=0.5, density=True)
_, bins, _ = plt.hist(samples_stack, bins=50, alpha=0.75, density=True)
plt.grid(True)
plt.show()

# ## Fit one Gaussian distribution

# mu, sigma = stats.norm.fit(samples_stack)  # equal statement
mu = np.mean(samples_stack)
sigma = np.std(samples_stack)

prob_one_gauss = stats.norm.pdf(x=bins, loc=mu, scale=sigma)  # probability density function

plt.hist(samples_stack, bins=50, alpha=0.75, density=True)
plt.plot(bins, prob_one_gauss, color='r', linewidth=2)
plt.grid(True)
plt.show()


