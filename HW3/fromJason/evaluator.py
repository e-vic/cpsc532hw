from primitives import env as penv
from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
from pyrsistent import pmap,plist

def standard_env():
    "An environment with some Scheme standard procedures."
    env = pmap(penv)
    env = env.update({'alpha' : ''}) 

    return env



def evaluate(exp, env=None): #TODO: add sigma, or something

    if env is None:
        env = standard_env()

    #TODO:
    return    


def get_stream(exp):
    while True:
        yield evaluate(exp)


def run_deterministic_tests():
    
    for i in range(1,14):

        exp = daphne(['desugar-hoppl', '-i', '../../HW5/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = evaluate(exp)
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))
        
        print('FOPPL Tests passed')
        
    for i in range(1,13):

        exp = daphne(['desugar-hoppl', '-i', '../../HW5/programs/tests/hoppl-deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/hoppl-deterministic/test_{}.truth'.format(i))
        ret = evaluate(exp)
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    num_samples=1e4
    max_p_value = 1e-2
    
    for i in range(1,7):
        exp = daphne(['desugar-hoppl', '-i', '../../HW5/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(exp)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    



if __name__ == '__main__':
    
    run_deterministic_tests()
    run_probabilistic_tests()
    

    for i in range(1,4):
        print(i)
        exp = daphne(['desugar-hoppl', '-i', '../../HW5/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(evaluate(exp))        
