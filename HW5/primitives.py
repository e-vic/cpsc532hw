import torch
import torch.distributions as dist



class Normal(dist.Normal):
    
    def __init__(self, alpha, loc, scale):
        
        if scale > 20.:
            self.optim_scale = scale.clone().detach().requires_grad_()
        else:
            self.optim_scale = torch.log(torch.exp(scale) - 1).clone().detach().requires_grad_()
        
        
        super().__init__(loc, torch.nn.functional.softplus(self.optim_scale))
    
    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.loc, self.optim_scale]
        
    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """
        
        ps = [p.clone().detach().requires_grad_() for p in self.Parameters()]
         
        return Normal(*ps)
    
    def log_prob(self, x):
        
        self.scale = torch.nn.functional.softplus(self.optim_scale)
        
        return super().log_prob(x)

def push_addr(alpha, value):
    return alpha + value

class Env(dict):
    "An environment: a dict of {'var': val} pairs, with an outer Env."
    def __init__(self, parms=(), args=(), outer=None):
        self.update(zip(parms, args))
        self.outer = outer
    def find(self, var):
        "Find the innermost Env where var appears."
        return self if (var in self) else self.outer.find(var)

class Procedure(object):
    "A user-defined Scheme procedure."
    def __init__(self, parms, body, env):
        self.parms, self.body, self.env = parms, body, env
    def __call__(self, *args): 
        return eval(self.body, Env(self.parms, args, self.env))

def fn(expr):
    print('lambda function')
    # return lambda

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



env = {
           'normal' : Normal,
           'push-address' : push_addr,
           'fn': fn,
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
       }






