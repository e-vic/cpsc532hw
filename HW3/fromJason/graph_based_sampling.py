import torch
import torch.distributions as dist
import matplotlib.pyplot as plt
import seaborn as sns

from daphne import daphne
import numpy as np

from primitives import PRIMITIVES
from tests import is_tol, run_prob_test,load_truth

# Put all function mappings from the deterministic language environment to your
# Python evaluation context here:
# env = {'normal': dist.Normal,
#        'sqrt': torch.sqrt}
env = PRIMITIVES

def deterministic_eval(exp):
    "Evaluation function for the deterministic target language of the graph based representation."
    if type(exp) is list:
        op = exp[0]
        args = exp[1:]
        return env[op](*map(deterministic_eval, args))
    elif type(exp) in [int, float]:
        # We use torch for all numerical objects in our evaluator
        return torch.Tensor([float(exp)]).squeeze()
    elif type(exp) is torch.Tensor:
        return exp
    elif type(exp) is bool:
        return torch.tensor(exp)
    else:
        raise Exception("Expression type unknown.", exp)

def topological_sort(nodes, edges):
    result = []
    visited = {}
    def helper(node):
        if node not in visited:
            visited[node] = True
            if node in edges:
                for child in edges[node]:
                    helper(child)
            result.append(node)
    for node in nodes:
        helper(node)
    return result[::-1]

def plugin_parent_values(expr, trace):
    if type(expr) == str and expr in trace:
        return trace[expr]
    elif type(expr) == list:
        return [plugin_parent_values(child_expr, trace) for child_expr in expr]
    else:
        return expr

def sample_from_joint(graph):
    "This function does ancestral sampling starting from the prior."
    # TODO insert your code here
    """
    1. Run topological sort on V using V and A, resulting in an array of v's
    2. Iterate through sample sites of the sorted array, and save sampled results on trace dictionary using P and Y
    - If keyword is sample*, first recursively replace sample site names with trace values in the expression from P. Then, run deterministic_eval.
    - If keyword is observe*, put the observation value in the trace dictionary
    3. Filter the trace dictionary for things sample sites you should return
    """
    procs, model, expr = graph[0], graph[1], graph[2]
    nodes, edges, links, obs = model['V'], model['A'], model['P'], model['Y']
    sorted_nodes = topological_sort(nodes, edges)

    sigma = {}
    trace = {}
    for node in sorted_nodes:
        keyword = links[node][0]
        if keyword == "sample*":
            link_expr = links[node][1]
            link_expr = plugin_parent_values(link_expr, trace)
            dist_obj  = deterministic_eval(link_expr)
            trace[node] = dist_obj.sample()
        elif keyword == "observe*":
            trace[node] = obs[node]

    expr = plugin_parent_values(expr, trace)
    return deterministic_eval(expr), sigma, trace

def nested_search(key,val,exp): 
    length = len(exp)
    if type(exp) is list:
        for i in range(length):
            if type(exp[i]) is list:
                nested_search(key,val,exp[i])
            else: 
                if exp[i] == key:
                    exp[i] = val
    return exp

def child(key,exp): 
    length = len(exp)
    if type(exp) is list:
        for i in range(length):
            if type(exp[i]) is list:
                child(key,exp[i])
            else: 
                if exp[i] == key:
                    return True
    return False

def accept(x,x_sample,X0,Q):
    Xp = {**X0}
    Xp[x] = x_sample
    q = Q[x] # q is the same for both X and X'
    q_expr0 = ["observeS",q[1],X0[x]]
    q_expr1 = ["observeS",q[1],x_sample]
    log_alpha = deterministic_eval(q_expr0) - deterministic_eval(q_expr1)

    Vx = [x]
    for X_key in list(X0.keys()):
        if child(x,Q[X_key]):
            Vx.append(X_key)

    for v in Vx:
        v_expr1 = ["observeS",plugin_parent_values(Q[v][1],Xp),X0[v]]
        v_expr0 = ["observeS",plugin_parent_values(Q[v][1],X0),X0[v]]

        log_alpha = log_alpha + deterministic_eval(v_expr1)
        log_alpha = log_alpha - deterministic_eval(v_expr0)

    return torch.exp(log_alpha)

def gibbs_step(X,Q):
    for x in list(Q.keys()):
        q = Q[x]
        q = plugin_parent_values(q, X)
        x_sample = deterministic_eval(q)

        alpha = accept(x,x_sample,X,Q)
        u = torch.distributions.Uniform(0,1).sample()
        if bool(u < alpha):
            X[x] = x_sample

        return X



def gibbs(graph,S):
    procs, model, expr = graph[0], graph[1], graph[2]
    nodes, edges, links, obs = model['V'], model['A'], model['P'], model['Y']
    sorted_nodes = topological_sort(nodes, edges)

    full_output = sample_from_joint(graph)
    X0 = full_output[2]
    Q = {}
    Q_temp = { k : links[k] for k in set(links) - set(obs) }
    for q_key in sorted_nodes: # sort Q topologically
        if q_key in list(Q_temp.keys()):
            Q[q_key] = Q_temp[q_key]

    X = [{k : X0[k] for k in list(Q.keys())}]
    X_out = [deterministic_eval(plugin_parent_values(expr,X[0]))]


    for q_key in list(Q.keys()):
        q = Q[q_key]
        q = nested_search('sample*','sampleS',q)
        Q[q_key] = plugin_parent_values(q,obs)

    for s in range(1,S+1):
        X.append(gibbs_step({**X[s-1]},Q))
        X_out.append(deterministic_eval(plugin_parent_values(expr,X[s])))
    
    return X_out

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
    
    for i in range(1,13):
        #note: this path should be with respect to the daphne path!
        graph = daphne(['graph','-i','../CS532-HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = deterministic_eval(graph[-1])
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for graph {}'.format(ret,truth,graph))
        
        print('Test passed')
        
    print('All deterministic tests passed')



def run_probabilistic_tests():
    
    #TODO: 
    num_samples = 1e4
    max_p_value = 1e-4
    
    for i in range(1,7):
        #note: this path should be with respect to the daphne path!        
        graph = daphne(['graph', '-i', '../CS532-HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        stream = get_stream(graph)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        print(p_val > max_p_value)
    
    print('All probabilistic tests passed')    


def print_tensor(tensor):
    tensor = np.round(tensor.numpy(), decimals=3)
    print(tensor)
        

if __name__ == '__main__':
    

    # run_deterministic_tests()
    # run_probabilistic_tests()
    prog_name = 'MHinGibbs'
    S = 10000

    for i in range(4,5):
        print('Program ',str(i))
        graph = daphne(['graph','-i','../HW3/fromJason/programs/hw3_p{}.daphne'.format(i)])
        # print('graph is: ',str(graph))

        if prog_name == 'MHinGibbs':
            samples = gibbs(graph,S)

        else:
            samples, n = [], S
            for j in range(n):
                full_output = sample_from_joint(graph)
                sample = full_output[0]
                samples.append(sample)
        # print('output is: ',str(samples))

        print(f'\nExpectation of return values for program {i}:')
        try:
            L = len(samples[0])
            expectation = [None]*L
            variance = [None]*L
            for k in range(L):
                variance[k] =  np.var([samples[i][k] for i in range(S+1)])
                expectation[k] =  np.mean([samples[i][k] for i in range(S+1)])
        except:
            expectation = np.mean(samples)
            variance = np.var(samples)
        
        print('expectation after ',str(S),' samples is: ',str(expectation))
        print('variance after ',str(S),' samples is: ',str(variance))

        # if type(samples[0]) is list:
        #     expectation = [None]*len(samples[0])
        #     variance = [None]*len(samples[0])
        #     for j in range(S):
        #         for k in range(len(expectation)):
        #             if expectation[k] is None:
        #                 expectation[k] = [samples[j][k]]
        #             else:
        #                 expectation[k].append(samples[j][k])
        #     for k in range(len(expectation)):
        #         print_tensor(sum(expectation[k])/S)
        #         variance[k] =  np.var([samples[i][k] for i in range(S+1)])
        #     print('variance is: ',str(variance))
        # else:
        #     expectation = sum(samples)/S
        #     variance = np.var(samples)
        #     print_tensor(expectation)
        #     print('variance is: ',str(variance))
