import numpy as np
from scipy.stats import ks_1samp, gamma

class Gamma:
    def __init__(self, x):
        self.x = x
    
    def fit(self):
        self.alpha = gamma.fit(self.x)[0]
        self.beta = gamma.fit(self.x)[2]
    
    def set_distribution(self):
        self.fit()
        self.dist = gamma(a=self.alpha,scale=self.beta)

    def test(self):
        self.set_distribution()
        ks_statistic, p_value = ks_1samp(self.x, self.dist.cdf)
        return ks_statistic, p_value
