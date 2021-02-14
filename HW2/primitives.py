import torch

# dist_env = {'normal': dist.Normal}

def vector(*vals):
    print('vector is called')
    output = [vals[i] for i in range(len(vals))]
    return torch.Tensor(output)

def put(*vals):
    print('put is called')
    vec = vals[0]
    print('vec type is: ',str(type(vec)))
    if isinstance(vec,dict):
        index = float(vals[1])
    else:
        index = int(vals[1])
    value = vals[2]
    print('value type is: ',str(type(value)))

    vec[index] = value
    return vec

def sampleS(dist):
    print('sample* is called')
    # takes in distribution type variable
    return dist.sample()

def hashmap(*vals):
    print('hashmap is called')
    print('input is: ',str(vals))
    # keys = [vals[i][0] for i in range(len(vals))]
    # values = [vals[i][1:] for i in range(len(vals))]
    keys = [vals[i] for i in range(0,len(vals),2)]
    values = [vals[i] for i in range(1,len(vals),2)]
    output = {float(keys[i]): values[i] for i in range(len(keys))}
    return output


def get(*vals):
    print('get is called')
    obj = vals[0]
    index = vals[1]
    return obj[index]
    # if isinstance(obj,dict):
    #     return obj[vals]
    # else:

def append(*vals):
    vec = vals[0]
    value = vals[1]
    return torch.cat(vec,value)

def let(vals): # not used
    print('let is called')
    v = vals[0] # will be a string?
    print('binding name is: ',v)
    e1 = vals[1] # will be a value?
    e2 = vals[2] # will be a string representing an expression?

    v_index = e2.index(v)
    e2[v_index] = e1
    return e2

def primitif(*vals):
    print('if called')
    logical = vals[0]
    true_value = vals[1]
    false_value = vals[2]

    print('logical type:',str(type(logical)))

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
    length = len(exp)
    if type(exp) is list:
        for i in range(length):
            if type(exp[i]) is list:
                print('type is list')
                nested_search(key,val,exp[i])
            else: 
                print('checking for key')
                print(exp[i])
                if exp[i] == key:
                    print('key found')
                    exp[i] = val
    return exp

            