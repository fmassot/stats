import numpy as np

from scipy.optimize import minimize
from scipy.stats.distributions import norm

from solver import Solver
from utils import get_log_likelihood

class GaussianMixPDF(object):
    def __init__(self, params):
        mu = params.get('mu')
        sigma = params.get('sigma')
        p = np.array(params.get('p'))
        p = np.array(p.tolist() + [1.-sum(p)])

        normals_pdf = [ (lambda obs: p_*norm(mu_, sigma_).pdf(obs)) for mu_, sigma_, p_ in zip(mu, sigma, p) ]
        
        def pdf(obs):
            return sum(map(lambda f: f(obs) , normals_pdf))

        self.pdf = pdf

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

def gaussian_mix_generator(mu, sigma, p, size=1000):
    gaussians = np.zeros((size, len(mu)))
    p = np.array(p)
    for id, (m, s, p) in enumerate(zip(mu, sigma, p)):
        gaussians[:,id] = p*np.random.normal(m, s, size=size)
    
    return gaussians.sum(1)

def test_max_likelihood():
    #simple test with a two gaussian mix
    params = { 'mu': [0., 1.],
               'sigma': [1., 5.], 
               'p': [0.3] }

    obs = gaussian_mix_generator(params['mu'], params['sigma'], params['p'])

    
    func_to_minimize = lambda params: -get_log_likelihood( GaussianMixPDF(params).pdf, obs)
    solver = Solver(func_to_minimize, minimize, preprocess=GaussianMixPDF.preprocess_func,
                    postprocess=GaussianMixPDF.postprocess_func)

    params0 = { 'mu': [0., 0.],
               'sigma': [1., 1.], 
               'p': [0.5] }
   
    bounds = ( (None, None), (None, None),
               (0., None), (0., None),
               (0., 1.) )
    res = solver.solve(params0)
    print res


