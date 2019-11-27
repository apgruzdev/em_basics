# # Imports

from dataclasses import dataclass
from typing import Iterable, Tuple, Union

from tqdm import tqdm

import numpy as np
import pandas as pd

from scipy import stats

from sklearn import metrics

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
dists_pars = [GaussParam(mu=5., sigma=2.), GaussParam(mu=15., sigma=3.)]
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

# ## QQ plot, "graphical" hypothesis testing

qs = [i/20 for i in range(1, 20)]
samples_qs = np.quantile(samples_stack, q=qs)
gauss_qs = stats.norm.ppf(q=qs, loc=mu, scale=sigma)  # percent point function

max_val = max(np.concatenate([samples_qs, gauss_qs]))
plt.plot([0, max_val], [0, max_val])
plt.scatter(x=gauss_qs, y=samples_qs, color='r')
plt.xlabel('theoretical quantiles')
plt.ylabel('sample quantiles')
plt.grid()
plt.show()

metrics.r2_score(y_true=samples_qs, y_pred=gauss_qs)

# ## Normality Tests

# ### Shapiro-Wilk Test

# H0: samples_stack came from a normally distributed population

alpha = 0.05  # fix before the test

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html
N = len(samples_stack)
assert N < 5000, 'For N > 5000 the W test statistic is accurate but the p-value may not be.'

_, p = stats.shapiro(samples_stack)

if p > alpha:
	print('Fail to reject H0 => normal distribution')
else:
	print('Reject H0 => "not" normal distribution')


# Here can be also applied to other normality tests

# ## EM implementation

class TwoGaussEM:
    """Expectation Maximization (EM) algorithm for two normal distributions"""
    def __init__(self, samples: Iterable[float],
                 gauss_first: GaussParam = GaussParam(mu=1, sigma=1), gauss_second: GaussParam = GaussParam(mu=2, sigma=1),
                 gamma: float = 0.5):
        self.samples = samples
        self.gauss_first = gauss_first
        self.gauss_second = gauss_second
        self.gamma = gamma  # hidden variable
        
        self.dim = len(self.samples)
        
        self.hist = {'log_likelihood': [], 
                     'gauss_first': [], 
                     'gauss_second': [],
                     'gamma': []}
        
    def partial_pdf(self, sample: Union[float, Iterable[float]]) -> Tuple[GaussParam, GaussParam]:
        """Probability Density Function (PDF) for each gaussian"""
        pdf_first = self.gamma * stats.norm.pdf(x=sample, loc=self.gauss_first.mu, scale=self.gauss_first.sigma)
        pdf_second = (1. - self.gamma) * stats.norm.pdf(x=sample, loc=self.gauss_second.mu, scale=self.gauss_second.sigma)
        return pdf_first, pdf_second
    
    def pdf(self, sample: Union[float, Iterable[float]]) -> GaussParam:
        """"Probability Density Function (PDF) for gaussian mixture model"""
        pdf_first, pdf_second = self.partial_pdf(sample)
        return self.gamma * pdf_first + (1. - self.gamma) * pdf_second
    
    def expect(self) -> Tuple[float, float]:
        """E-step"""
        pdf_first, pdf_second = self.partial_pdf(self.samples)
        denominator = pdf_first + pdf_second
        self.hist['log_likelihood'].append(np.sum(np.log(denominator)))  # logging

        w_first = pdf_first / denominator
        w_second = np.ones_like(w_first) - w_first

        return w_first, w_second
            
    def maximize(self, weights: Tuple[float, float]) -> None:
        """M-step"""
        w_first, w_second = weights
        
        denom_first = np.sum(w_first)
        denom_second = np.sum(w_second)
        
        self.gauss_first = GaussParam(mu=np.sum(w_first * self.samples) / denom_first, 
                                      sigma=np.sqrt(np.sum(w_first * (self.samples - self.gauss_first.mu)**2) / denom_first))
        self.gauss_second = GaussParam(mu=np.sum(w_second * self.samples) / denom_second, 
                                       sigma=np.sqrt(np.sum(w_second * (self.samples - self.gauss_second.mu)**2) / denom_second))
        self.hist['gauss_first'].append(self.gauss_first)  # logging
        self.hist['gauss_second'].append(self.gauss_second)  # logging
        
        self.gamma = denom_first / self.dim
        self.hist['gamma'].append(self.gamma)  # logging
        
    def one_iter(self) -> None:
        """Lounch one iteration of EM algorithm"""
        self.maximize(self.expect())
        
    def launch(self, iter_num: int) -> None:
        """Lounch one iteration of EM algorithm"""
        for it in tqdm(range(iter_num)):
            self.one_iter()


# launch
em = TwoGaussEM(samples=samples_stack)
em.launch(iter_num=20)

# results
em.gauss_first, em.gauss_second


