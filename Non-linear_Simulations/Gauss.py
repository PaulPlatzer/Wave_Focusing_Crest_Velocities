## Gaussian function ##

def gauss(x, *p):
    import numpy as np
    a, mu, sigma = p
    return a*np.exp(-(x-mu)**2/(2*sigma**2))