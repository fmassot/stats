import numpy as np

from functools import partial
from scipy.cluster.vq import kmeans2
from scipy.optimize import minimize
from scipy.stats.distributions import norm

from utils import get_log_likelihood
from solver import Solver

class GaussianMix(object):
    def __init__(self, params):
        mu = params.get('mu')
        sigma = params.get('sigma')
        p = np.array(params.get('p'))
        p = np.abs(p.tolist() + [1.-sum(p)])

        def normals_pdf(mu, sigma, p, obs):
            return sum( [ p_*norm(mu_, sigma_).pdf(obs) for mu_, sigma_, p_ in zip(mu, sigma, p) ] )
       
        #WARNING: not very clever to use this function to compute log likelyhood: log(exp) is not very accurate...
        self.pdf = partial(normals_pdf, mu, sigma, p)

        
        self.log_pdf = None

    @classmethod
    def postprocess_func(kls, params):
        mu = params.get('mu')
        sigma = params.get('sigma')
        p = params.get('p')
        return mu + sigma + p

    @classmethod
    def preprocess_func(kls, x):
        r = (len(x)+1) / 3
        return { 'mu': x[:r],
                 'sigma': x[r:r*2],
                 'p': x[r*2:]
               }

def gaussian_mix_generator(mu, sigma, p, size=10000):
    gaussians = np.zeros((size, len(mu)))
    p = np.array(p)
    p = np.array(p.tolist() + [1.-sum(p)])
    multinomial = np.random.multinomial(1, p, size=size)
    for id, (m, s) in enumerate(zip(mu, sigma)):
        gaussians[:, id] += np.random.normal(m, s, size=size)
    
    return (gaussians * multinomial).sum(1)

def test_max_likelihood(params=None, params0=None, method="L-BFGS-B"):
    #simple test with a two gaussian mixture
    if not params:
        params = { 'mu': [0., 5.],
                   'sigma': [0.01, 0.01], 
                   'p': [0.5] }

    obs = gaussian_mix_generator(params['mu'], params['sigma'], params['p'])

    """
    def penalize(params):
        penalization = 0
        if True in (np.array(params['p'])<0.01):
            penalization += 10000000.
        if True in (np.array(params['sigma'])<0.01):
            penalization += 10000000.
        return penalization
    """

    func_to_minimize = lambda params:  - get_log_likelihood( GaussianMix(params).pdf, obs)
    solver = Solver(func_to_minimize, minimize, preprocess=GaussianMix.preprocess_func,
                    postprocess=GaussianMix.postprocess_func)

    bounds = ( (None, None), (None, None),
               (0., None), (0., None),
               (0., 1.) )

    #the initial point is set by kmeans
    k = len(params['mu'])
    kmeans_res = kmeans2(obs, k)
    mu0 = kmeans_res[0]
    sigma0 = np.zeros(k)
    p0 = np.zeros(k)

    for id in range(k):
        sigma0[id] = obs[np.where(kmeans_res[1]==id)].std()
        p0[id] = np.where(kmeans_res[1]==id)[0].size * 1. / obs.size

    params0 = { 'mu': mu0.tolist(),
                'sigma': sigma0.tolist(),
                'p': p0[:-1].tolist()
                }
    print "initial point", params0

    res = solver.solve(params0, method=method, bounds=bounds)
    print "final point", res

    print "function value for the correct point", func_to_minimize(params)
    print "function value for the initial point", func_to_minimize(params0)
    print "function value for the final point", func_to_minimize(res)


