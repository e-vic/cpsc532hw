# CPSC 532W Homework 1
# Emma Hansen
# January 28, 2021

# --------------------------- #
using Images, FileIO, ImageMagick, Random, LinearAlgebra, Colors, StatsBase, MAT

# Question 3
p_cloudy = [0.5,0.5]
cloudy_vals = [true,false]

p_sprinkler = [[0.1,0.9],[0.5,0.5]]
sprinkler_vals = [[true,false],[true,false]]

p_rain = [[0.8,0.2],[0.2,0.8]]
rain_vals = [[true,false],[true,false]]

p_wetgrass = [[0.99,0.01],[0.9,0.1],[0.9,0.1],[0,1]]
wetgrass_vals = [[true,false],[true,false],[true,false],[true,false]]
p_wetgrass2 = [[[0.99,0.01],[0.9,0.1]],[[0.9,0.1],[0,1]]]

## a)
## compute joint:
p = zeros(2,2,2,2) #c,s,r,w
for c=1:2
    for s=1:2
        for r=1:2
            for w=1:2
                p[c,s,r,w] = p_cloudy[c]*p_sprinkler[c][s]*p_rain[c][r]*p_wetgrass2[s][r][w]
            end
        end
    end
end

wetgrass_true = sum(p[:,:,:,1])
wetgrass_false = sum(p[:,:,:,2])

wetgrassANDcloudy_true = sum(p[1,:,:,1])

cloudyGIVENwetgrass_true = (wetgrassANDcloudy_true/p_cloudy[1])*p_cloudy[1]/wetgrass_true

print("The probability of it being cloudy given that the grass is wet is: ",string(cloudyGIVENwetgrass_true),".")

## b) Ancestral Sampling and Rejection
N = 50000 # number of samples
n = 1
wetgrass_samples = zeros(N)
cloudy_samples = zeros(N)
rejections = 0
while n <= N
    cloudy = sample(cloudy_vals,aweights(p_cloudy))

    if cloudy == true
        cloudy_ind = 1
    else
        cloudy_ind = 2
    end

    sprinkler = sample(sprinkler_vals[cloudy_ind,:][1],aweights(p_sprinkler[cloudy_ind,:][1]))

    rain = sample(rain_vals[cloudy_ind,:][1],aweights(p_rain[cloudy_ind,:][1]))

    if (sprinkler == true) & (rain == true) 
        row = 1
    elseif (sprinkler == true) & (rain == false)
        row = 2
    elseif (sprinkler == false) & (rain == true)
        row = 3
    elseif (sprinkler == false) & (rain == false)
        row = 4
    else 
        print("error? cloudy is: ",cloudy,", sprinkler is: ",sprinkler,", rain is: ",rain)
    end

    wetgrass = sample(wetgrass_vals[row,:][1],aweights(p_wetgrass[row,:][1]))
    #wetgrass_samples[n] = wetgrass

    if wetgrass == true
        cloudy_samples[n] = cloudy
    else
        rejections = rejections + 1
    end
    n = n+1
end

p_cloudyandwetgrass = sum(cloudy_samples)/N # probability of the grass being wet and it's cloudy
p_cloudy_wetgrass = p_cloudyandwetgrass/wetgrass_true

print("The probability that it is cloudy given that the grass is wet is: ",string(p_cloudy_wetgrass))
print("Percentage of rejected samples is: ",string(100*rejections/N))

# c) Gibbs Sampling

N = 80000
trim = 5000
c_sample = rand(1:2) # 1- true, 2 - false
r_sample = rand(1:2)
s_sample = rand(1:2)
p_CSR = zeros(3,N)
for n=1:N
    probNEWc = p[:,s_sample,r_sample,1]
    probNEWc = probNEWc/sum(probNEWc)
    c_sample = sample([1,2],aweights(probNEWc))

    probNEWs = p[c_sample,:,r_sample,1]
    probNEWs = probNEWs/sum(probNEWs)
    s_sample = sample([1,2],aweights(probNEWs))

    probNEWr = p[c_sample,s_sample,:,1]
    probNEWr = probNEWr/sum(probNEWr)
    r_sample = sample([1,2],aweights(probNEWr))

    p_CSR[:,n] = [c_sample s_sample r_sample]
end

p_CSR_trimmed = p_CSR[:,trim:end]
p_cloudyGIVENwetgrass_gibbs = sum(p_CSR_trimmed[1,p_CSR_trimmed[1,:].==1])/(N-trim)

# Question 5
bagofwords = matread("HW1/bagofwords_nips.mat")
WS = bagofwords["WS"]
DS = bagofwords["DS"]

WO = matread("HW1/words_nips.mat")["WO"][:,1]
titles = matread("HW1/titles_nips.mat")["titles"][:,1]

alphabet_size = size(WO)

document_assignment = DS
words = WS

#subset data, EDIT THIS PART ONCE YOU ARE CONFIDENT THE MODEL IS WORKING
#PROPERLY IN ORDER TO USE THE ENTIRE DATA SET
# 
#
#
# NEED TO MAKE JULIA COMPATIBLE BELOW HERE
words = words[document_assignment < 100]
document_assignment  = document_assignment[document_assignment < 100]

n_docs = document_assignment.max() + 1

#number of topics
n_topics = 20

#initial topic assigments
topic_assignment = np.random.randint(n_topics, size=document_assignment.size)

#within document count of topics
doc_counts = np.zeros((n_docs,n_topics))

for d in range(n_docs):
    #histogram counts the number of occurences in a certain defined bin
    doc_counts[d] = np.histogram(topic_assignment[document_assignment == d], bins=n_topics, range=(-0.5,n_topics-0.5))[0]

#doc_N: array of size n_docs count of total words in each document, minus 1
doc_N = doc_counts.sum(axis=1) - 1

#within topic count of words
topic_counts = np.zeros((n_topics,alphabet_size))

for k in range(n_topics):
    w_k = words[topic_assignment == k]

    topic_counts[k] = np.histogram(w_k, bins=alphabet_size, range=(-0.5,alphabet_size-0.5))[0]



#topic_N: array of size n_topics count of total words assigned to each topic
topic_N = topic_counts.sum(axis=1)

#prior parameters, alpha parameterizes the dirichlet to regularize the
#document specific distributions over topics and gamma parameterizes the 
#dirichlet to regularize the topic specific distributions over words.
#These parameters are both scalars and really we use alpha * ones() to
#parameterize each dirichlet distribution. Iters will set the number of
#times your sampler will iterate.
alpha = None
gamma = None 
iters = None 


jll = []
for i in range(iters):
    jll.append(joint_log_lik(doc_counts,topic_counts,alpha,gamma))
    
    prm = np.random.permutation(words.shape[0])
    
    words = words[prm]   
    document_assignment = document_assignment[prm]
    topic_assignment = topic_assignment[prm]
    
    topic_assignment, topic_counts, doc_counts, topic_N = sample_topic_assignment(
                                topic_assignment,
                                topic_counts,
                                doc_counts,
                                topic_N,
                                doc_N,
                                alpha,
                                gamma,
                                words,
                                document_assignment)
                        
jll.append(joint_log_lik(doc_counts,topic_counts,alpha,gamma))

plt.plot(jll)
