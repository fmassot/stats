

class Solver(object):
    def __init__(self, func, solver_method, preprocess=None, postprocess=None):
        #function which convert a vector of parameters into set of parameters of
        #the function to be minimize
        if not preprocess:
            preprocess = lambda x:x
        self.preprocess = preprocess
        #inverse of preprocess
        if not postprocess:
            postprocess = lambda x:x
        self.postprocess = postprocess
        #function in input of a solver method
        self.func = func
        self.solver_method = solver_method

    def solve(self, params0, **kwargs):
        def func_to_minimize(x):
            return self.func(self.preprocess(x))
        self.res = self.solver_method(func_to_minimize, self.postprocess(params0), full_output=True, **kwargs)
        return self.preprocess(self.res[0])


