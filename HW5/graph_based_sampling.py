import torch
import torch.distributions as dist
import matplotlib.pyplot as plt

from daphne import daphne
import numpy as np
import copy

from primitives import PRIMITIVES
from tests import is_tol, run_prob_test,load_truth
from distributions import *

import time

dev_cpu = torch.device("cpu")
# device = torch.device("cuda:0")

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
    sigma['q'] = {}
    trace = {}
    for node in sorted_nodes:
        keyword = links[node][0]
        if keyword == "sample*":
            link_expr = links[node][1]
            link_expr = plugin_parent_values(link_expr, trace)
            dist_obj  = deterministic_eval(link_expr)
            sigma['q'][node] = dist_obj.make_copy_with_grads()
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
    # print('sample is: ',str(x))
    Xp = {**X0}
    Xp[x] = x_sample
    q = Q[x] # q is the same for both X and X'
    q = plugin_parent_values(q, X0)
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
        # print('q in gibbs step is: ',str(q))
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

    # print('sorted vars: ',str(sorted_nodes))

    # print('graph is: ', str(links))

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


def bbvi_eval(node,expr,sigma,trace,t):
    """
    copied from above sample_from_joint, with updated sample and observe statements
    """
    keyword = expr[0]
    # print('in bbvi_eval')
    # print('full expression is: ',str(expr))
    if keyword == "sample*":
        link_expr = expr[1]
        link_expr = plugin_parent_values(link_expr, trace)
        dist_obj0  = deterministic_eval(link_expr)
        dist_obj = dist_obj0.make_copy_with_grads()
        learning_rate = 1/(t+10)

        if  node not in list(sigma['q'].keys()): #once initialized it never gets reinitialised
            sigma['q'][node] = dist_obj
            # print('torch optimisation parameters: ',str(torch.optim.Adam(dist_obj.Parameters(), lr=1e-2)))
            sigma['opt'][node] = torch.optim.Adam(dist_obj.Parameters(), lr=learning_rate)
            # sigma['opt'][node] = torch.optim.SGD(dist_obj.Parameters(), lr=1/(t+10))
        else:
            sigma['opt'][node] = torch.optim.Adam(sigma['q'][node].Parameters(), lr=learning_rate)
            # sigma['opt'][node] = torch.optim.SGD(sigma['q'][node].Parameters(), lr=1/(t+10))
        
        c = sigma['q'][node].sample()
        lp = 1*sigma['q'][node].log_prob(c)
        # print('lp is: ',str(lp))
        lp.backward()
        # print('what is v in dist_obj.params? ',str([v for v in sigma['q'][node].Parameters()]))
        # print('gradient of log prob ', str([v.grad for v in sigma['q'][node].Parameters()]))
        
        trace[node] = c
        # print('gradients in eval are: ',str([v.grad for v in sigma['q'][node].Parameters()]))
        grads = [v.grad for v in sigma['q'][node].Parameters()]
        # print('gradient ',str(grads))
        # sigma['G'][node] = torch.tensor([*grads])
        sigma['G'][node] = [*grads]
        # print('first part: ',str(dist_obj.log_prob(c)))
        sigma['logW'] = sigma['logW'] + (dist_obj.log_prob(c) - lp)
        # print('logW: ',str(sigma['logW']))

        return c, sigma

    elif keyword == "observe*":
        e1 = expr[1]
        e1 = plugin_parent_values(e1, trace)
        e2 = expr[2]
        e2 = plugin_parent_values(e2, trace)
        p = deterministic_eval(e1)
        c = deterministic_eval(e2)
        obs = p.log_prob(c)
        sigma['logW'] = sigma['logW'] + obs
        # trace[node] = obs
        trace[node] = c

        return c, sigma

def elbo_grad(Gl,logWl):
    nodes = list(set([list(G.keys())[i] for G in Gl for i in range(len(list(G.keys())))]))
    g_hat = {}
    L = len(logWl)
    for node in nodes:
        F = []
        for l in range(L):
            if node in list(Gl[l].keys()):
                F.append(torch.multiply(Gl[l][node],logWl[l]))
            else:
                F.append(torch.tensor([0.,0.]))
                Gl[l][node] = torch.tensor([0.,0.])
        b_hat = sum([np.cov(F[l].detach(),Gl[l][node].detach())[0,1] for l in range(L)])/sum([np.var([Gl[l][node][i].detach().numpy() for l in range(L)]) for i in range(len(Gl[l][node]))])
        # b_hat = 0

        temp = [np.multiply(-1*b_hat,Gl[l][node]) for l in range(L)]
        Ftemp = [torch.add(F[i],temp[i]) for i in range(len(F))]
        sumFtemp = torch.stack(Ftemp,dim=0).sum(dim=0)
        g_hat[node] = torch.divide(sumFtemp,L)
    return g_hat

def bbvi(T,L,graph):
    procs, model, expr = graph[0], graph[1], graph[2]
    nodes, edges, links, obs = model['V'], model['A'], model['P'], model['Y']
    sorted_nodes = topological_sort(nodes, edges)

    _, sigma,temp = sample_from_joint(graph) 
    # USE SAMPLE FROM JOINT TO TURN EXPR INTO PRE-EVALUATED THING THAT CAN JUST BE SAMPLED/OBSERVED LIKE IN ALG 14, 
    # SO RECOMPUTNG DOESN'T HAVE TO HAPPEN

    print('expression is: ',str(expr))
    # print('links are: ',str(links))
    # print('sigma q is: ',str(sigma['q']))
    # print('sorted nodes are: ',str(sorted_nodes))

    expr2 = plugin_parent_values(expr,temp)
    temp_outputs = deterministic_eval(expr2)
    try:
        num_outputs = len(temp_outputs)
    except:
        num_outputs = 1

    sigma['q'] = {}
    sigma['logW'] = 0
    sigma['G'] = {}
    trace = {}
    r = [None]*T
    logW = [None]*T

    for t in range(T):
        rl = []
        Gl = [None]*L
        # Gl = []
        logWl = []
        sigma['opt'] = {}
        for l in range(L):
            sigma['logW'] = 0
            # Gl[l] = {}
            for node in sorted_nodes:
                # print('node is: ',str(node))
                _,sigma = bbvi_eval(node,links[node],sigma,trace,t)
                # print('Gl right after bbvi eval: ',str(Gl))
                # if node in list(sigma['q'].keys()):
                #     Gl[l][node] = [*grads]
            # print('gradients are: ',str(sigma['G']))
            r_expr = plugin_parent_values(expr,trace)
            rl.append(deterministic_eval(r_expr))
            Gl[l] = copy.deepcopy(sigma['G'])
            # print('Gl updated? ',str(Gl))
            logWl.append(*[sigma['logW']])
            # print('trace updated? ',str(trace))
        
        # print('Gl is: ', str(Gl))
        # print('sigma opt is: ',str(sigma['opt']))
        # print('trace is: ',str(trace))

        # NEED TO UPDATE PARAMETER GRADIENTS WITH THE ONES FROM ELBO GRAD

        # USE THIS FOR BHAT
        # g_hat = elbo_grad(Gl,logWl)
        # # print('g_hat is: ',str(g_hat))
        # for node in sigma['opt'].keys():
        #     p = 0
        #     for param in sigma['q'][node].Parameters():
        #         param.grad = g_hat[node][p] # update the gradients to be g_hat
        #         p = p + 1
        #     sigma['opt'][node].step()
        #     for param in sigma['q'][node].Parameters():
        #         param.grad = 0*param.grad # need to zero the gradients since zero_grad isn't working with custom gradients
        #         p = p + 1

        # USE THIS FOR NO BHAT
        for node in sigma['opt'].keys():
            sigma['opt'][node].step()
            sigma['opt'][node].zero_grad()

        # print('does q change? ',str(sigma['q']))
        r[t] = [*rl]
        logW[t] = [*logWl]
    return r,logW,sigma['q'],num_outputs

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
    # prog_name = 'MHinGibbs'
    prog_name = 'BBVI'
    S = 50 # number of samples
    L = 300 # number of gradient steps

    for i in range(3,4):
        print('Program ',str(i))
        graph = daphne(['graph','-i','../HW3/fromJason/programs/hw4_{}.daphne'.format(i)])
        # print('graph is: ',str(graph))

        tic = time.perf_counter()
        if prog_name == 'MHinGibbs':
            samples = gibbs(graph,S)
        elif prog_name == 'BBVI':
            full_output = bbvi(L,S,graph)
            samples = full_output[0]
            logW = full_output[1]
            Q = full_output[2]
            num_outputs = full_output[-1]
            # print('unsorted samples are: ',str(samples))
            # print('full output is: ',str(full_output))
            # print('number of outputs are: ', str(num_outputs))
            print('final distribution is: ',str(Q))
            if num_outputs == 1:
                # converged_samples = [[samples[s][-1] for s in range(S)]]
                converged_samples = [[samples[-1]]]
            else:
                # converged_samples = [[samples[s][-1][k] for s in range(S)] for k in range(num_outputs)]
                converged_samples = [[samples[-1][s][k] for s in range(S)] for k in range(num_outputs)]
            
            # print('converged samples are: ', str(converged_samples))
        else:
            samples, n = [], S
            for j in range(n):
                full_output = sample_from_joint(graph)
                sample = full_output[0]
                samples.append(sample)
        toc = time.perf_counter()
        # print('samples are: ',str(samples))
        print('elapsed time is: ',str(toc-tic))

        # print(f'\nExpectation of return values for program {i}:')
        try:
            if prog_name != 'BBVI':
                N = len(samples[0])
                expectation = [None]*N
                variance = [None]*N
                for k in range(N):
                    variance[k] =  np.var([samples[i][k] for i in range(S+1)])
                    expectation[k] =  np.mean([samples[i][k] for i in range(S+1)])
            else:
                expectation = [None]*num_outputs
                variance = [None]*num_outputs
                for k in range(num_outputs):
                    variance[k] = np.var(converged_samples[k])
                    expectation[k] = np.mean(converged_samples[k])
        except:
            if i != 4:
                expectation = np.mean(samples)
                variance = np.var(samples)
        
        print('expectation after ',str(S),' samples is: ',str(expectation))
        print('variance after ',str(S),' samples is: ',str(variance))
        print(str(L),' gradient steps were used')

        #LogW plot
        figW,axW = plt.subplots()
        axW.plot([logW[k][-1] for k in range(L)])
        axW.set_ylabel('logW')
        plt.show()

        # histograms
        try:
            if prog_name != 'BBVI':
                N = len(samples[0])
                figcols = 2
                figrows = int(np.ceil(N/figcols))
                fig = plt.figure(figsize=(5,2*figrows))
                grid = plt.GridSpec(figrows, figcols, figure=fig, hspace=0.35, wspace=0.2)

                axes = {}
                k = 0
                for n in range(figrows):
                    for m in range(figcols):
                        axes[str(n)+str(m)] = fig.add_subplot(grid[n,m])
                        k = k+1
                        if k >= N:
                            break

                k = 0
                for n in range(figrows):
                    for m in range(figcols):
                        axes[str(n)+str(m)].hist([float(val[k]) for val in samples])
                        k = k+1
                        if k >= N:
                            break

                plt.show()
            else: 
                if num_outputs == 1:
                    figcols = 1
                else:
                    figcols = 2
                figrows = int(np.ceil(num_outputs/figcols))
                fig = plt.figure(figsize=(5,2*figrows))
                grid = plt.GridSpec(figrows, figcols, figure=fig, hspace=0.35, wspace=0.2)

                axes = {}
                k = 0
                for n in range(figrows):
                    for m in range(figcols):
                        axes[str(n)+str(m)] = fig.add_subplot(grid[n,m])
                        k = k+1
                        if k >= num_outputs:
                            break

                k = 0
                for n in range(figrows):
                    for m in range(figcols):
                        axes[str(n)+str(m)].hist(converged_samples[k])
                        k = k+1
                        if k >= num_outputs:
                            break

                plt.show()
        except:
            fig, ax = plt.subplots()
            ax.hist([float(val) for val in samples])
            plt.show()

        # sample trace
        if (i == 1 or i == 2) and prog_name != 'BBVI':
            try:
                N = len(samples[0])
                figcols = 2
                figrows = int(np.ceil(N/figcols))
                fig = plt.figure(figsize=(5,2*figrows))
                grid = plt.GridSpec(figrows, figcols, figure=fig, hspace=0.35, wspace=0.2)

                axes = {}
                k = 0
                for n in range(figrows):
                    for m in range(figcols):
                        axes[str(n)+str(m)] = fig.add_subplot(grid[n,m])
                        k = k+1
                        if k >= N:
                            break

                k = 0
                for n in range(figrows):
                    for m in range(figcols):
                        axes[str(n)+str(m)].plot([float(val[k]) for val in samples])
                        k = k+1
                        if k >= N:
                            break

                plt.show()
            except:
                fig, ax = plt.subplots()
                ax.plot([float(val) for val in samples])
                plt.show()


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
