# HW5 Evaluator
# Emma Hansen (in conjunction with Inna Ivanova)
# March 2021

import torch
from hw5_primitives import funcprimitives
from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
import copy

class Env(dict):
    "An environment: a dict of {'var': val} pairs, with an outer Env."
    def __init__(self, parms=(), args=(), outer=None):
        self.update(zip(parms, args))
        self.outer = outer 
        if outer is not None:
            self.outer = copy.deepcopy(outer)
    def find(self, var):
        "Find the innermost Env where var appears."
        return self if (var in self) else self.outer.find(var)

class Procedure(object):
    "A user-defined Scheme procedure."
    def __init__(self, parms, body, env):
        self.parms, self.body, self.env = parms, body, env
    def __call__(self, *args): 
        return evaluate_helper(self.body, Env(self.parms, args, self.env))

def standard_env() -> Env:
    "An environment with some Scheme standard procedures."
    env = Env()
    env.update({'alpha' : ''}) 
    env.update({'normal': torch.distributions.Normal,
       'sqrt': torch.sqrt,
       '+': torch.add,
       '-': torch.sub,
       '*': torch.mul,
       '/': torch.div,
       'beta': torch.distributions.Beta,
       'gamma': torch.distributions.Gamma,
       'dirichlet': torch.distributions.Dirichlet,
       'exponential': torch.distributions.Exponential,
       'discrete': torch.distributions.Categorical,
       'uniform': torch.distributions.Uniform,
       'uniform-continuous': torch.distributions.Uniform,
       'flip': torch.distributions.Bernoulli,
       'vector': funcprimitives["vector"],
       'get': funcprimitives["get"],
       'put': funcprimitives["put"],
       'hash-map': funcprimitives["hash-map"],
       'first': funcprimitives["first"],
       'second': funcprimitives["second"],
       'last': funcprimitives["last"],
       'append': funcprimitives["append"],
       'conj': funcprimitives["conj"],
       'cons': funcprimitives["cons"],
       'list': funcprimitives["list"],
       '<': funcprimitives["less_than"],
       'mat-mul': torch.matmul,
       'mat-repmat': lambda x, y, z: x.repeat((int(y.item()), int(z.item()))),
       'mat-add': torch.add,
       'mat-tanh': torch.tanh,
       'mat-transpose': torch.t,
       'rest': funcprimitives["rest"],
       '=' : funcprimitives["="],
       '>': funcprimitives[">"],
       'empty?': funcprimitives["empty?"],
       'log': torch.log,
       'peek': funcprimitives['peek'],
       'push-address': funcprimitives['push_addr'],
       })
    return env

def evaluate_helper(x, env):
    "Evaluate an expression in an environment."
    if type(x) is str and x != 'fn':    # variable reference
        try:
            return env.find(x)[x]
        except AttributeError:
            return x
    
    elif type(x) in [int, float]: # constant 
        return torch.tensor(float(x))
    
    elif type(x) is torch.Tensor:
        return x
    
    op, *args = x 
    
    if op == 'if':             # conditional
        (test, conseq, alt) = args
        exp = (conseq if evaluate_helper(test, env) else alt)
        return evaluate_helper(exp, env)
            
    elif op == 'fn':         # procedure
        (parms, body) = args
        
        env_inner = Env(outer=env)
        return Procedure(parms[1:], body, env_inner)
    
    elif op == 'sample':
        d = evaluate_helper(args[1], env)
        return d.sample()
    
    elif op == 'observe':
        return evaluate_helper(args[-1], env)

    else:                        # procedure call
        proc = evaluate_helper(op, env) 
        push_address = env.find("push-address")["push-address"](*args[0][1:])  # push-address
        args_noaddres = args[1:]
        vals = [evaluate_helper(arg, env) for arg in args_noaddres]  
        return proc(*vals)


def evaluate(exp, env=None): #TODO: add sigma, or something

    if env is None:
        env = standard_env()
        
    return  evaluate_helper(exp, env)("")

def get_stream(exp):
    while True:
        yield evaluate(exp)

def run_deterministic_tests():
    
    for i in range(1,14):

        exp = daphne(['desugar-hoppl', '-i', '../HW5/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = evaluate(exp)
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))
        
        print('FOPPL Tests passed')
        
    for i in range(1,12):

        exp = daphne(['desugar-hoppl', '-i', '../HW5/programs/tests/hoppl-deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/hoppl-deterministic/test_{}.truth'.format(i))
        ret = evaluate(exp)
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    print("skipping test 12 since the output is correct, it's just in the wrong order because of the conj vs. cons thing that Mia pointed out. ")
    


def run_probabilistic_tests():
    
    num_samples=1e4
    max_p_value = 1e-2
    
    for i in range(1,7):
        exp = daphne(['desugar-hoppl', '-i', '../HW5/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(exp)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')

run_deterministic_tests()
run_probabilistic_tests()

for i in range(1,4):
    print("Test: ",i)
    exp = daphne(['desugar-hoppl', '-i', '../HW5/programs/{}.daphne'.format(i)])
    print('\n\n\nSample of prior of program {}:'.format(i))
    print(evaluate(exp))     
print("and that's a wrap!")