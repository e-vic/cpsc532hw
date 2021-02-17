import torch
import torch.distributions as dist

# dist_env = {'normal': dist.Normal}

def vector(*vals):
    print('vector is called')
    print('values are: ',str(vals))
    if issubclass(type(vals[0]),torch.distributions.distribution.Distribution):
        output = [vals[i] for i in range(len(vals))]
        return output
    else:
        try:
            length = len(vals[0])
            # print('length is ',str(length))
        except:
            # print('length is 1')
            length = 1

        if length > 1:
            # output = [val[i] for val in vals for i in range(len(val))]
            # print('vector is creating a matrix')
            output = torch.cat(vals,1)
            return output
        else:
            output = [vals[i] for i in range(len(vals))]
            # print(torch.Tensor(output).resize_((len(output),1)))
            return torch.Tensor(output).resize_((len(output),1))
        # print(type(output[0]))
        

def put(*vals):
    # print('put is called')
    vec = vals[0]
    # print('vec type is: ',str(type(vec)))
    if isinstance(vec,dict):
        index = float(vals[1])
    else:
        index = int(vals[1])
    value = vals[2]
    # print('value type is: ',str(type(value)))

    vec[index] = value
    return vec

def sampleS(dist):
    print('sample* is called')
    print('input to sample is: ',str(dist))
    # takes in distribution type variable
    return dist.sample()

def hashmap(*vals):
    # print('hashmap is called')
    # print('input is: ',str(vals))
    # keys = [vals[i][0] for i in range(len(vals))]
    # values = [vals[i][1:] for i in range(len(vals))]
    keys = [vals[i] for i in range(0,len(vals),2)]
    values = [vals[i] for i in range(1,len(vals),2)]
    output = {float(keys[i]): values[i] for i in range(len(keys))}
    return output


def get(*vals):
    # print('get is called')
    obj = vals[0]
    index = vals[1]
    
    if isinstance(obj,dict):
        return obj[index]
    else:
        # print('index is: ',str(int(index)))
        return obj[int(index)]

def get_eval(*vals):
    # print('get is called')
    obj = vals[0]
    index = vals[1]
    
    if isinstance(obj,dict):
        if type(list(obj.keys())[0]) is str:
            return obj[str(index)]
        else:
            return obj[float(index)]
    else:
        # print('index is: ',str(int(index)))
        return obj[int(index)]

def append(*vals):
    vec = vals[0]
    value = vals[1]
    return torch.cat(vec,value)

def let(vals): # not used
    print('let is called')
    v = vals[0] # will be a string?
    # print('binding name is: ',v)
    e1 = vals[1] # will be a value?
    e2 = vals[2] # will be a string representing an expression?

    v_index = e2.index(v)
    e2[v_index] = e1
    return e2

def primitif(*vals):
    # print('if called')
    logical = vals[0]
    true_value = vals[1]
    false_value = vals[2]

    # print('logical type:',str(type(logical)))

    if logical == True:
        return true_value
    else:
        return false_value

def leq(*vals):
    if vals[0] < vals[1]:
        return True
    else:
        return False

def geq(*vals):
    if vals[0] > vals[1]:
        return True
    else:
        return False

def nested_search(key,val,exp): # this is my let function actually
    # print('let/nested search is called')
    length = len(exp)
    if type(exp) is list:
        for i in range(length):
            if type(exp[i]) is list:
                # print('type is list')
                nested_search(key,val,exp[i])
            else: 
                # print('checking for key')
                # print(exp[i])
                if exp[i] == key:
                    # print('key found')
                    exp[i] = val
    return exp

def observeS(*vals):
    # print('observe is called')
    # print('vals are: ',str(vals))
    distribution = vals[0]
    rand_var = vals[1]
    output = torch.exp(distribution.log_prob(rand_var))
    # print('output is: ',str(output))
    return output

def sort_variables(V_ordered,V_parent,A):
    V_ordered = [] + V_parent
    for key in V_parent:
        if key in list(A.keys()):
            V_child = A[key]
            if V_child not in V_ordered:
                V_ordered = V_ordered + V_child
                V_ordered = V_ordered + sort_variables(V_ordered,V_child,A)
    return V_ordered

def transpose(*vals):
    # print('transpose is called')
    # print('vals are: ',str(vals))
    if len(vals) == 3 and isinstance(vals[1],int):
        return torch.transpose(vals)
    else:
        # vals = vals.resize_((list(vals.shape)[0],1))
        return torch.transpose(vals[0],0,1)

def repmat(*vals):
    # print('repmat is called')
    # print('input is: ',str(vals))
    tensor = vals[0]
    reps = tuple([int(val) for val in vals[1:]])
    return tensor.repeat(reps)

def matmul(*vals):
    # print('matmul is called')
    size1 = list(vals[0].shape)
    size2 = list(vals[1].shape)

    if size1[1] != size2[0]:
        if size1[0] == size2[0]:
            return torch.matmul(torch.transpose(vals[0],0,1),vals[1])
        else:
            return torch.matmul(vals[0],torch.transpose(vals[1],0,1))
    else:
        return torch.matmul(vals[0],vals[1])

def discrete(*vals):
    # print('discrete is called')
    # print(vals)
    length = max(list(vals[0].shape))
    probs = vals[0].resize_((length))
    return dist.Categorical(probs)

def first(vals):
    return vals[0]

def last(vals):
    return vals[-1]

def append(*vals):
    return torch.cat((vals[0],vals[1].resize_((1,1))),0)