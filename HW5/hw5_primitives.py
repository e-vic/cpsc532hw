# HW5 Primitives

import torch


def vector(*arg):
    if len(arg) == 0:
        return torch.tensor([])
    # general case
    try:
        return torch.stack(arg, dim=0)
    
    # for concatenation of many vectors
    except RuntimeError:
        dim = len(arg[0].shape) - 1
        return torch.cat(arg, dim=dim)
    
    # for distribution objects
    except TypeError:
        return list(arg)

def get(v, i):
    if type(i) is str:
        return v[i]
    return v[int(i.item())]

def put(v, i, c):
    if type(i) is str:
        v[i] = c
    else:
        v[int(i.item())] = c
    return v

def first(v):
    return v[0]

def second(v):
    return v[1]

def last(v):
    return v[-1]

def append(v, c):
    return torch.cat((v, c.unsqueeze(dim=0)), dim=0)

# def conj(v, c):
#     return torch.cat((c.unsqueeze(dim=0), v), dim=0)

def hashmap(*v):
    hm = {}
    i = 0
    while i < len(v):
        if type(v[i]) is str:
            hm[v[i]] = v[i+1]
        else:
            hm[v[i].item()] = v[i+1]
        i+=2
    return hm

def less_than(*args):
    return args[0] < args[1]

def rest(v):
    return v[1:]

def l(*arg):
    return list(arg)

def cons(x, l):
    return [x] + l  

def equal(x, y):
    return torch.tensor(x.item() == y.item())

def and_fn(x, y):
    return x and y

def or_fn(x, y):
    return x or y

def dirac(x):
    # approximate with a normal distribution but with very small std
    return torch.distributions.Normal(x, 0.001)

def greater_than(x, y):
    return x > y

def empty(v):
    return len(v) == 0

def peek(v):
    return v[-1]

def push_addr(alpha, value):
    return alpha + value

funcprimitives = {
    "vector": vector,
    "get": get,
    "put": put,
    "first": first,
    "last": last,
    "append": append,
    "hash-map": hashmap,
    "less_than": less_than,
    "second": second,
    "rest": rest,
    "conj": append,
    "list": l,
    "cons": cons,
    "=": equal,
    "and": and_fn,
    "or": or_fn,
    "dirac": dirac,
    ">": greater_than,
    "empty?": empty,
    "peek": peek,
    "push_addr": push_addr,
}

