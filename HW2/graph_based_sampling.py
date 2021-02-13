import torch
import torch.distributions as dist
import numpy as np

from daphne import daphne

# from primitives import funcprimitives #TODO
from primitives import *
from tests import is_tol, run_prob_test,load_truth

# Put all function mappings from the deterministic language environment to your
# Python evaluation context here:
env = {'normal': dist.Normal,
       'sqrt': torch.sqrt,
       '+': torch.add,
       '-': torch.sub,
       '/': torch.div,
       'vector': vector,
       'put': put,
       'sample*': sampleS,
       'beta': dist.Beta,
       'exponential': dist.Exponential,
       'hash-map': hashmap,
       'get': get}#,
    #    }


def deterministic_eval(exp):
    "Evaluation function for the deterministic target language of the graph based representation."
    if type(exp) is list:
        op = exp[0]
        print('in det eval, op is: ',str(op))
        args = exp[1:]
        print('args are: ',str(args))
        # print('type of args is: ',str(type(args)))
        # print('return value is: ',str(env[op](*map(deterministic_eval, args))))
        if op=='hash-map':
            args =  [args[i][j] for i in range(np.shape(args)[0]) for j in range(np.shape(args)[1])]
        print('type of args is: ',str(type(args)))

        return env[op](*map(deterministic_eval, args))
    elif type(exp) is int or type(exp) is float:
        # We use torch for all numerical objects in our evaluator
        return torch.tensor(float(exp))
    else:
        raise("Expression type unknown.", exp)


def sample_from_joint(graph):
    "This function does ancestral sampling starting from the prior."
    # TODO insert your code here
    V = graph[1]['V']
    A = graph[1]['A']
    P = graph[1]['P']
    Y = graph[1]['Y']
    
    # print(V)
    # print(P)

    for var in V:
        print(P[var])
        # print(type(P[var]))
        var_sample = deterministic_eval(P[var])
        # print(var_dist)
            

    return var_sample


def get_stream(graph):
    """Return a stream of prior samples
    Args: 
        graph: json graph as loaded by daphne wrapper
    Returns: a python iterator with an infinite stream of samples
        """
    while True:
        yield sample_from_joint(graph)




#Testing:

def run_deterministic_tests():
    
    for i in range(1,14):
        #note: this path should be with respect to the daphne path!
        graph = daphne(['graph','-i','../HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        print('graph is: ', str(graph))
        print('last entry of graph is: ',str(graph[-1]))
        ret = deterministic_eval(graph[-1])
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for graph {}'.format(ret,truth,graph))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    #TODO: 
    # num_samples=1e4
    num_samples=1
    max_p_value = 1e-4
    
    for i in range(1,7):
        #note: this path should be with respect to the daphne path!        
        graph = daphne(['graph', '-i', '../HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        # print(graph)
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        print('graph is: ', str(graph))
        print('last entry of graph is: ',str(graph[-1]))
        stream = get_stream(graph)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    


        
        
if __name__ == '__main__':
    

    run_deterministic_tests()
    # run_probabilistic_tests()




    # for i in range(1,5):
    #     graph = daphne(['graph','-i','../CS532-HW2/programs/{}.daphne'.format(i)])
    #     print('\n\n\nSample of prior of program {}:'.format(i))
    #     print(sample_from_joint(graph))    

    