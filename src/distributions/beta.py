import numpy as np
from scipy.stats import ks_1samp, beta

class Beta:
    def __init__(self, x):
        self.x = x
    
    def fit(self):
        self.alpha = beta.fit(self.x)[0]
        self.beta = beta.fit(self.x)[1]
    
    def set_distribution(self):
        self.fit()
        self.dist = beta(a=self.alpha,b=self.beta)

    def test(self):
        self.set_distribution()
        ks_statistic, p_value = ks_1samp(self.x, self.dist.cdf)
        return ks_statistic, p_value
