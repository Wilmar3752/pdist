import numpy as np
from scipy.stats import ks_1samp
from scipy.stats import norm

class Normal:
    def __init__(self, x):
        self.x = x
    
    def fit(self):
        self.mu, self.sigma = norm.fit(self.x)
    
    def set_distribution(self):
        self.fit()
        self.dist = norm(loc=self.mu, scale=self.sigma)

    def test(self):
        self.set_distribution()
        ks_statistic, p_value = ks_1samp(self.x, self.dist.cdf)
        return ks_statistic, p_value
