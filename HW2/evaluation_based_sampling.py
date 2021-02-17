from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
import torch
import torch.distributions as dist
from primitives import *
import copy

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

def evaluate_program(ast):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """
    # Algorithm 6 goes here?
    e = copy.deepcopy(ast[0])
    print('expression is: ',str(e))
    try: 
        l = copy.deepcopy(ast[1])
    except:
        l = {}


    if type(e) is int or type(e) is float:
        print('operation: none, just a number')
        print('number is: ',str(e))
        return torch.tensor(float(e))

    if type(e) is str:
        if e in list(l.keys()):
            return l[e]
        else:
            return e

    else:
        if e[0] == 'sample' and issubclass(type(e[1]),torch.distributions.distribution.Distribution):
            return sampleS(e[1])

        elif e[0] == 'observe' and issubclass(type(e[1]),torch.distributions.distribution.Distribution):
            return observeS(e[1:])

        elif e[0] == 'if' and type(e[1]) is bool:
            # not sure, come back to this
            print('if block, need to write this still')

        elif e[0] == 'let':
            print('let block, come back to this')

        else:
            c = [None]*len(e)
            for i in range(len(e)):
                print('that c setting loop')
                c[i] = evaluate_program([e[i]])
                print('c is: ',str(c))
            if type(e[0]) is int or type(e[0]) is float or (type(e[0]) is torch.tensor and list(e[0].shape)==[]):
                return c

            elif e[0] in list(env.keys()):
                if e[0] == '+' or e[0] == '*' or e[0] == '/':
                    print('e[0] is ',str(e[0]))
                    print('e[1] is',str(e[1]))
                    print('e[2] is',str(e[2]))
                    return env[e[0]](evaluate_program([e[1]]),evaluate_program([e[2]]))
                else:
                    print('user defined functions')
                    print('function is: ',str(e[0]))
                    print('args are: ',str(e[1:]))
                    # return env[e[0]](*map(evaluate_program,[e[1:]]))
                    if len(c[1:])==1:
                        return env[e[0]](c[1])
                    else:
                        print('c args are')
                        print(c[1:])
                        return env[e[0]](c[1:])



    # return None


def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast)
    


def run_deterministic_tests():
    
    for i in range(1,14):
        #note: this path should be with respect to the daphne path!
        # ast = daphne(['desugar', '-i', '../CS532-HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        ast = daphne(['desugar', '-i', '../HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        print('ast is: ',str(ast))
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = evaluate_program(ast)
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,ast))
        
        print('!Test ',str(i),' passed!')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    num_samples=1e4
    max_p_value = 1e-4
    
    for i in range(1,7):
        #note: this path should be with respect to the daphne path!        
        # ast = daphne(['desugar', '-i', '../CS532-HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        ast = daphne(['desugar', '-i', '../HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(ast)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    

        
if __name__ == '__main__':

    run_deterministic_tests()
    
    run_probabilistic_tests()


    for i in range(1,5):
        # ast = daphne(['desugar', '-i', '../CS532-HW2/programs/{}.daphne'.format(i)])
        ast = daphne(['desugar', '-i', '../HW2/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(ast)
        print(evaluate_program(ast)[0])