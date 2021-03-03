from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from primitives import PRIMITIVES
from collections.abc import Iterable
import time


def evaluate_program(ast,prog_name='prior_sampling',prog_args=[]):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """
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
    if prog_name == 'prior_sampling':
        return eval(ast[-1], {}, {})
    elif prog_name == 'importance_sampling':
        print('Importance Sampling')
        L = prog_args
        importance_out = []
        for l in range(L):
            r_l, sigma_l = eval(ast[-1],{'logW': 0},{})
            importance_out.append([r_l,sigma_l['logW']])

        return importance_out





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


def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast)
    


def run_deterministic_tests():
    
    for i in range(1,14):
        #note: this path should be with respect to the daphne path!
        ast = daphne(['desugar', '-i', '../HW3/fromJason/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret, sig = evaluate_program(ast)
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            #raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,ast))
            print('return value {} is not equal to truth {} for exp {}'.format(ret,truth,ast))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    num_samples=1e4
    max_p_value = 1e-4
    
    for i in range(1,7):
        #note: this path should be with respect to the daphne path!        
        ast = daphne(['desugar', '-i', '../HW3/fromJason/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(ast)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    


def print_tensor(tensor):
    tensor = np.round(tensor.numpy(), decimals=3)
    print(tensor)

        
if __name__ == '__main__':
    # run_deterministic_tests()
    # run_probabilistic_tests()

    for i in range(1,3):
        print('Program ',str(i))
        ast = daphne(['desugar', '-i', '../HW3/fromJason/programs/hw3_p{}.daphne'.format(i)])

        # samples, n = [], 1
        # for j in range(n):
        #     sample = evaluate_program(ast)[0]
        #     samples.append(sample)
        
        prog_name = 'importance_sampling'
        L = 2
        tic = time.perf_counter()
        samples = evaluate_program(ast,prog_name,L)
        toc = time.perf_counter()
        print('elapsed time: ',str(toc-tic))
        print('samples are: ', str(samples))

        if prog_name == 'importance_sampling':
            W_sum = sum([torch.exp(val[1]) for val in samples])
            expectation = sum([val[0]*torch.exp(val[1]) for val in samples])/W_sum
            variance = sum([torch.square(val[0])*torch.exp(val[1]) for val in samples])/W_sum - torch.square(expectation)
            print('expectation after ',str(L), 'samples is: ',str(expectation))
            print('variance is: ',str(variance))


            # histograms
            try:
                N = len(samples[0][0])
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
                        axes[str(n)+str(m)].hist([float(val[0][k]) for val in samples])
                        k = k+1
                        if k >= N:
                            break

                plt.show()
            except:
                fig, ax = plt.subplots()
                ax.hist([float(val[0]) for val in samples])
                plt.show()

            # if i == 1 or i == 2:
            #     try:
            #         N = len(samples[0][0])
            #         figcols = 2
            #         figrows = int(np.ceil(N/figcols))
            #         fig = plt.figure(figsize=(5,2*figrows))
            #         grid = plt.GridSpec(figrows, figcols, figure=fig, hspace=0.35, wspace=0.2)

            #         axes = {}
            #         k = 0
            #         for n in range(figrows):
            #             for m in range(figcols):
            #                 axes[str(n)+str(m)] = fig.add_subplot(grid[n,m])
            #                 k = k+1
            #                 if k >= N:
            #                     break

            #         k = 0
            #         for n in range(figrows):
            #             for m in range(figcols):
            #                 axes[str(n)+str(m)].plot([float(val[0][k]) for val in samples])
            #                 k = k+1
            #                 if k >= N:
            #                     break

            #         plt.show()
            #     except:
            #         fig, ax = plt.subplots()
            #         ax.plot([float(val[0]) for val in samples])
            #         plt.show()



        else:
            print(f'\nExpectation of return values for program {i}:')
            if type(samples[0]) is list:
                expectation = [None]*len(samples[0])
                for j in range(L):
                    for k in range(len(expectation)):
                        if expectation[k] is None:
                            expectation[k] = [samples[j][k]]
                        else:
                            expectation[k].append(samples[j][k])
                for k in range(len(expectation)):
                    print_tensor(sum(expectation[k])/L)
            else:
                expectation = sum(samples)/L
                print_tensor(expectation)

"""
Plot Code
plottype = 'eval'

# 1
plt.hist(samples)
plt.savefig(f'figures/p1{plottype}.png')

# 2
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.hist([a[0] for a in samples])
ax2.hist([a[1] for a in samples])
plt.savefig(f'figures/p2{plottype}.png')

# 3
fig, axs = plt.subplots(3,6)
png = [axs[i//6,i%6].hist([a[i] for a in samples]) for i in range(17)]
plt.tight_layout()
plt.savefig(f'figures/p3{plottype}.png')

# 4
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,2))
mean1 = [np.mean([s[0].flatten()[j] for s in samples]) for j in range(len(samples[0][0].flatten()))]
var1  = [np.var([s[0].flatten()[j] for s in samples]) for j in range(len(samples[0][0].flatten()))]
sns.heatmap(np.array(mean1).reshape(2,5),ax=ax1,annot=True,fmt="0.3f")
sns.heatmap(np.array(var1).reshape(2,5),ax=ax2,annot=True,fmt="0.3f")
ax1.set_title('Marginal Mean')
ax2.set_title('Marginal Variance')
plt.tight_layout()
plt.savefig(f'figures/p41{plottype}.png')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,2))
mean2 = [np.mean([s[1].flatten()[j] for s in samples]) for j in range(len(samples[0][1].flatten()))]
var2  = [np.var([s[1].flatten()[j] for s in samples]) for j in range(len(samples[0][1].flatten()))]
sns.heatmap(np.array(mean2).reshape(2,5),ax=ax1,annot=True,fmt="0.3f")
sns.heatmap(np.array(var2).reshape(2,5),ax=ax2,annot=True,fmt="0.3f")
ax1.set_title('Marginal Mean')
ax2.set_title('Marginal Variance')
plt.tight_layout()
plt.savefig(f'figures/p42{plottype}.png')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,7))
mean3 = [np.mean([s[2].flatten()[j] for s in samples]) for j in range(len(samples[0][2].flatten()))]
var3  = [np.var([s[2].flatten()[j] for s in samples]) for j in range(len(samples[0][2].flatten()))]
sns.heatmap(np.array(mean3).reshape(10,10),ax=ax1,annot=True,fmt="0.3f")
sns.heatmap(np.array(var3).reshape(10,10),ax=ax2,annot=True,fmt="0.3f")
ax1.set_title('Marginal Mean')
ax2.set_title('Marginal Variance')
plt.tight_layout()
plt.savefig(f'figures/p43{plottype}.png')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,2))
mean4 = [np.mean([s[3].flatten()[j] for s in samples]) for j in range(len(samples[0][3].flatten()))]
var4  = [np.var([s[3].flatten()[j] for s in samples]) for j in range(len(samples[0][3].flatten()))]
sns.heatmap(np.array(mean4).reshape(2,5),ax=ax1,annot=True,fmt="0.3f")
sns.heatmap(np.array(var4).reshape(2,5),ax=ax2,annot=True,fmt="0.3f")
ax1.set_title('Marginal Mean')
ax2.set_title('Marginal Variance')
plt.tight_layout()
plt.savefig(f'figures/p44{plottype}.png')
"""