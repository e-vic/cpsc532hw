from primitives import env as penv
from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
from pyrsistent import pmap,plist

def standard_env():
    "An environment with some Scheme standard procedures."
    env = pmap(penv)
    env = env.update({'alpha' : ''}) 

    return env



def evaluate(ast, env=None): #TODO: add sigma, or something

    if env is None:
        env = standard_env()

    #TODO:
    PROCS = {}
    for i in range(len(ast)-1):
        proc = ast[i]
        proc_name, proc_arg_names, proc_expr = proc[1], proc[2], proc[3]
        PROCS[proc_name] = (proc_arg_names,proc_expr)

    def eval(expr, sigma, scope):
        if is_const(expr, scope):
            if type(expr) in [int, float]:
                expr = torch.Tensor([expr]).squeeze()
            return expr, sigma
        elif is_var(expr, scope):
            return scope[expr], sigma
        elif is_let(expr, scope):
            var_name, sub_expr, final_expr = expr[1][0], expr[1][1], expr[2]
            var_value, sigma = eval(sub_expr, sigma, scope)
            return eval(final_expr, sigma, {**scope, var_name: var_value})
        elif is_if(expr,scope):
            cond_expr, true_expr, false_expr = expr[1], expr[2], expr[3]
            cond_value, sigma = eval(cond_expr, sigma, scope)
            if cond_value:
                return eval(true_expr, sigma, scope)
            else:
                return eval(false_expr, sigma, scope)
        elif is_sample(expr,scope):
            dist_expr = expr[1]
            dist_obj, sigma = eval(dist_expr,sigma,scope)
            return dist_obj.sample(), sigma
        elif is_observe(expr,scope):
            dist_expr, obs_expr = expr[1], expr[2]
            dist_obj, sigma = eval(dist_expr,sigma,scope)
            obs_value, sigma = eval(obs_expr,sigma,scope)
            sigma['logW'] = sigma['logW'] + dist_obj.log_prob(obs_value)
            return obs_value, sigma
        else:
            proc_name = expr[0]
            consts = []
            for i in range(1,len(expr)):
                const, sigma = eval(expr[i],sigma,scope)
                consts.append(const)
            if proc_name in PROCS:
                proc_arg_names, proc_expr = PROCS[proc_name]
                new_scope = {**scope}
                for i, name in enumerate(proc_arg_names):
                    new_scope[name] = consts[i]
                return eval(proc_expr, sigma, new_scope)
            else:
                return PRIMITIVES[proc_name](*consts), sigma
    
    return eval(ast[-1], {}, {})

def is_const(expr, scope):
    return type(expr) not in [tuple,list,dict] and expr not in PRIMITIVES and expr not in scope
def is_var(expr, scope):
    return type(expr) not in [tuple,list,dict] and expr in scope
def is_let(expr, scope):
    return expr[0] == "let"
def is_if(expr, scope):
    return expr[0] == "if"
def is_sample(expr, scope):
    return expr[0] == "sample"
def is_observe(expr, scope):
    return expr[0] == "observe"


def get_stream(exp):
    while True:
        yield evaluate(exp)


def run_deterministic_tests():
    
    for i in range(1,14):


        exp = daphne(['desugar-hoppl-noaddress', '-s', '../HW5/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        print('expression is: ',str(exp))
        ret = evaluate(exp)
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))
        
        print('FOPPL Test '+str(i)+' passed')
        
    for i in range(1,13):

        exp = daphne(['desugar-hoppl', '-i', '../HW5/programs/tests/hoppl-deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/hoppl-deterministic/test_{}.truth'.format(i))
        ret = evaluate(exp)
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))
        
        print('HOPPL Test '+str(i)+' passed')
        
    print('All deterministic tests passed')
    


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
        print('Probabilistic Test '+str(i)+' passed')
    
    print('All probabilistic tests passed')    



if __name__ == '__main__':
    
    run_deterministic_tests()
    run_probabilistic_tests()
    

    for i in range(1,4):
        print(i)
        exp = daphne(['desugar-hoppl', '-i', '../HW5/programs/hw5_{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(evaluate(exp))        
