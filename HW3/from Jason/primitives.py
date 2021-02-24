from copy import deepcopy
import torch
from collections.abc import Iterable


def add(a,b):
    return torch.add(a,b)
def subtract(a,b):
    return torch.subtract(a,b)
def multiply(a,b):
    return torch.multiply(a,b)
def divide(a,b):
    return torch.divide(a,b)
def gt(a,b):
    return a>b
def lt(a,b):
    return a<b
def eq(a,b):
    return a==b
def sqrt(a):
    return torch.sqrt(a)
def tanh(a):
    return torch.tanh(a)
def first(data):
    return data[0]
def second(data):
    return data[1]
def rest(data):
    return data[1:]
def last(data):
    return data[-1]
def nth(data, index):
    return data[index]
def conj(data, el):
    if len(el.shape) == 0: el = el.reshape(1)
    return torch.cat([data, el], dim=0)
def cons(data, el):
    if len(el.shape) == 0: el = el.reshape(1)
    return torch.cat([el, data], dim=0)
def vector(*args):
    # sniff test: if what is inside isn't int,float,or tensor return normal list
    if type(args[0]) not in [int, float, torch.Tensor]:
        return [arg for arg in args]
    # if tensor dimensions are same, return stacked tensor
    if type(args[0]) is torch.Tensor:
        sizes = list(filter(lambda arg: arg.shape == args[0].shape, args))
        if len(sizes) == len(args):
            return torch.stack(args)
        else:
            return [arg for arg in args]
    raise Exception(f'Type of args {args} could not be recognized.')
def hashmap(*args):
    result, i = {}, 0
    while i<len(args):
        key, value  = args[i], args[i+1]
        if type(key) is torch.Tensor:
            key = key.item()
        result[key] = value
        i += 2
    return result
def get(struct, index):
    if type(index) is torch.Tensor:
        index = index.item()
    if type(struct) in [torch.Tensor, list, tuple]:
        index = int(index)
    return struct[index]
def put(struct, index, value):
    if type(index) is torch.Tensor:
        index = int(index.item())
    if type(struct) in [torch.Tensor, list, tuple]:
        index = int(index)
    result = deepcopy(struct)
    result[index] = value
    return result
def bernoulli(p, obs=None):
    return torch.distributions.Bernoulli(p)
def beta(alpha, beta, obs=None):
    return torch.distributions.Beta(alpha,beta)
def normal(mu, sigma):
    return torch.distributions.Normal(mu, sigma)
def uniform(a, b):
    return torch.distributions.Uniform(a, b)
def exponential(lamb):
    return torch.distributions.Exponential(lamb)
def discrete(vector):
    return torch.distributions.Categorical(vector)
def transpose(tensor):
    return tensor.T
def repmat(tensor, size1, size2):
    if type(size1) is torch.Tensor: size1 = int(size1.item())
    if type(size2) is torch.Tensor: size2 = int(size2.item())
    return tensor.repeat(size1, size2)
def matmul(t1, t2):
    return t1.matmul(t2)

PRIMITIVES = {
    "+": add,
    "-": subtract,
    "*": multiply,
    "/": divide,
    ">": gt,
    "<": lt,
    "==": eq,
    "sqrt": sqrt,
    "first": first,
    "second": second,
    "rest": rest,
    "last": last,
    "nth": nth,
    "append": conj,
    "conj": conj,
    "cons": cons,
    "vector": vector,
    "hash-map": hashmap,
    "list": list,
    "get": get,
    "put": put,
    "bernoulli": bernoulli,
    "beta": beta,
    "normal": normal,
    "uniform": uniform,
    "exponential": exponential,
    "discrete": discrete,
    "mat-transpose": transpose,
    "mat-add": add,
    "mat-tanh": tanh,
    "mat-repmat": repmat,
    "mat-mul": matmul,
    "if": lambda cond, v1, v2: v1 if cond else v2 # for graph based sampling
}
