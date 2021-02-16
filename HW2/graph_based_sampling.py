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
       '*': torch.mul,
       'vector': vector,
       'put': put,
       'sample*': sampleS,
       'beta': dist.Beta,
       'exponential': dist.Exponential,
       'hash-map': hashmap,
       'get': get,
       'uniform': dist.Uniform,
       'if': primitif,
       '<': leq,
       '>': geq,
       'let': nested_search,
       'observe*': observeS,
       'discrete': discrete,
       'mat-transpose': transpose,
       'mat-tanh': torch.tanh,
       'mat-add': torch.add,
       'mat-mul': matmul,
       'mat-repmat': repmat}


def deterministic_eval(exp):
    "Evaluation function for the deterministic target language of the graph based representation."
    if type(exp) is list:
        op = exp[0]
        # print('in det eval, op is: ',str(op))
        args = exp[1:]
        # print('args are: ',str(args))
        # print('type of args is: ',str(type(args)))
        # print('type of args is: ',str(type(args)))
        # print('return value is: ',str(env[op](*map(deterministic_eval, args))))
        if op=='hash-map':
            args =  [args[i][j] for i in range(np.shape(args)[0]) for j in range(np.shape(args)[1])]
        # print('type of args is: ',str(type(args)))

        return env[op](*map(deterministic_eval, args))
    elif type(exp) is int or type(exp) is float:
        # We use torch for all numerical objects in our evaluator
        return torch.tensor(float(exp))
    else:
        raise("Expression type unknown.", exp)


def sample_from_joint(graph):
    "This function does ancestral sampling starting from the prior."
    # Need to order random vars based on parent-child structure given in A
    V = graph[1]['V']
    A = graph[1]['A']
    P = graph[1]['P']
    Y = graph[1]['Y']

    V_parent = []
    children = list(set([list(A.values())[i][j] for i in range(len(list(A.values()))) for j in range(len(list(A.values())[i]))]))
    # print('V is: ',str(V))
    # print('parent-child relationships: ',str(A))
    for var in V:
        if var in list(A.keys()):
            if var not in children:
                V_parent = V_parent + [var]
            
    V_ordered = []
    ordered = sort_variables([],V_parent,A)
    for i in range(len(ordered)):
        if ordered[i] not in V_ordered:
            V_ordered = V_ordered + [ordered[i]]      
    
    # print(V)
    # print(P)
    var_samples = {}
    for var in V_ordered:
        if 'current_keys' in locals():
            # print('current_keys exists')
            # print(var,' probability is: ',str(P[var]))
            new_P = P[var]
            for key_name in current_keys:
                # print('Updating with prev. random var values')
                # print('keyname is: ',str(key_name))
                # print('key value is: ',str(var_samples[key_name]))
                # new_P = ['let', key_name, var_samples[key_name], new_P]
                new_P = nested_search(key_name,float(var_samples[key_name]), new_P)
                # print('new statement is: ',str(new_P))

            var_samples[var] = deterministic_eval(new_P)
            # print(var,' value is: ',str(var_samples[var]))
            
            current_keys = list(var_samples.keys())
        else:
            # print(var,' probability is: ',str(P[var]))
            var_samples[var] = deterministic_eval(P[var])
            # print(var,' value is: ',str(var_samples[var]))

            current_keys = list(var_samples.keys())
        
    # print('Samples are: ',str(var_samples))

    out_key = graph[2]
    if graph[2][0] == 'vector':
        for key_name in current_keys:
            out_key = nested_search(key_name,float(var_samples[key_name]), out_key)
        print('outkey is: ',str(out_key))
        return  deterministic_eval(out_key)
    else:
        out_key = graph[2]
        print('outkey is: ',str(out_key))
        return var_samples[out_key]
    
    


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
        if i != 4:

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
    

    # run_deterministic_tests()
    # run_probabilistic_tests()




    for i in range(1,5):
        graph = daphne(['graph','-i','../HW2/programs/{}.daphne'.format(i)])
        print('graph is:',str(graph))
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(sample_from_joint(graph))    

    