from evaluator import evaluate
import torch
import numpy as np
import json
import sys
import matplotlib.pyplot as plt
import time


def run_until_observe_or_end(res):
    cont, args, sigma = res
    res = cont(*args)
    while type(res) is tuple:
        if res[2]['type'] == 'observe':
            return res
        cont, args, sigma = res
        res = cont(*args)

    res = (res, None, {'done' : True}) #wrap it back up in a tuple, that has "done" in the sigma map
    return res

def resample_particles(particles, log_weights):
    # n_weights = np.exp([*log_weights.detach()]) / np.sum(np.exp([*log_weights.detach()]))
    w_dist = torch.distributions.Categorical(logits=torch.tensor(log_weights))

    ind_list = []
    for i in range(len(particles)):
        ind_list.append(w_dist.sample())

    new_particles = [particles[i] for i in ind_list]
    # buildl index map with categorial distribution
    # logsumexp of log weights

    logZ = torch.logsumexp(torch.tensor(log_weights),0)/len(particles)

    return logZ, new_particles



def SMC(n_particles, exp):

    particles = []
    weights = []
    logZs = []
    output = lambda x: x

    for i in range(n_particles):

        res = evaluate(exp, env=None)('addr_start', output)
        logW = 0.


        particles.append(res)
        weights.append(logW)

    #can't be done after the first step, under the address transform, so this should be fine:
    done = False
    smc_cnter = 0
    while not done:
        # print('In SMC step {}, length of Zs: '.format(smc_cnter), len(logZs))
        for i in range(n_particles): #Even though this can be parallelized, we run it serially
            res = run_until_observe_or_end(particles[i])
            if 'done' in res[2]: #this checks if the calculation is done
                particles[i] = res[0]
                if i == 0:
                    done = True  #and enforces everything to be the same as the first particle
                    address = ''
                else:
                    if not done:
                        raise RuntimeError('Failed SMC, finished one calculation before the other')
            else:
                cont, args, sigma = res
                particles[i] = res 
                weights[i] = sigma['logW']
                # address check: what you have in particle compared to what you have in res
                original_address = particles[0][2]['alpha']
                current_address = sigma['alpha']

                if original_address == current_address:
                    pass
                else:
                    print('did not pass address check :(')
                    raise
                # pass #TODO: check particle addresses, and get weights and continuations

        if not done:
            #resample and keep track of logZs
            logZn, particles = resample_particles(particles, weights)
            logZs.append(logZn)
        smc_cnter += 1
    logZ = sum(logZs)
    return logZ, particles


if __name__ == '__main__':
    #for i in range(1,5):
    for i in range(1,2):
        print("\n\n Program ",str(i))
        with open('programs/{}.json'.format(i),'r') as f:
            exp = json.load(f)
        # n_p = [1, 10, 100, 1000, 10000, 100000]
        n_p = [1,10,100,1000,10000]

        fig = plt.figure(figsize=(10,6))
        grid = plt.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
        axes = {}
        j = 0
        for n in range(2):
            for m in range(3):
                axes[str(j)] = fig.add_subplot(grid[n,m])
                j = j + 1
        j = 0
        for n in n_p:
            print("\nnumber of particles: ",str(n))

            tic = time.perf_counter()
            logZ, particles = SMC(n, exp)
            toc = time.perf_counter()
            print('elapsed time is: ',str(toc-tic))

            values = torch.stack(particles)
            if n == 1:
                if i != 3:
                    values = torch.tensor([float(values[k]) for k in range(len(values))])
            if type(logZ) == int:
                Z = np.exp(float(logZ))
            else:
                Z = np.exp(logZ)

            print("Marginal probability/evidence: ", Z)
            if i == 3:
                num_variables = len(values[0])
                expectation = [None]*num_variables
                variance = [None]*num_variables
                values_binned = [None]*num_variables
                for k in range(num_variables):
                    variable_vals = [values[j][k] for j in range(n)]
                    variance[k] =  np.var(variable_vals)
                    expectation[k] =  np.mean(variable_vals)
                    variable_bins = np.digitize(variable_vals, range(3))
                    values_binned[k] = [np.sum(variable_bins==bin_val) for bin_val in range(1,4)]
                    
                print("the mean is ", expectation)
                print("the variance is", variance)
                
            else:
                float_vals = [float(values[k]) for k in range(len(values))]
                print("the mean is ", np.mean(float_vals))
                print("the variance is", np.var(float_vals))
            
            if i ==3:
                cax = axes[str(j)].imshow(values_binned,aspect='auto',cmap='Blues')
                cbar = fig.colorbar(cax,ax = axes[str(j)])
                # cbar.ax.set_yticklabels(['0','1','2'])
                axes[str(j)].set_ylabel('Variables')
                axes[str(j)].set_xticks([0,1,2])
                axes[str(j)].set_yticks(ticks=np.arange(0,num_variables,2))
                axes[str(j)].set_yticks(ticks=np.arange(0,num_variables),minor=True)
                axes[str(j)].grid(which='both',axis='y')
            else:
                axes[str(j)].hist(values)
            
            axes[str(j)].set_title( "{} particles".format(n) )
            j=j+1
        plt.show()
        fig.savefig('program'+str(i)+'_hist.png')