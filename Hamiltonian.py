import math
import warnings

import numpy as np

from chainer import backend
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import optimizer
from chainer import Variable
import chainer.functions as F

_default_hyperparam = optimizer.Hyperparameter()
_default_hyperparam.epsilon = 0.1
_default_hyperparam.delta = 0.95


class HamiltonianRule(optimizer.UpdateRule):
    
    """ Update rule of Hamiltonian Gradient Descent.
    
    sample ver.
    """
    
    _kernel = None
    
    def __init__(self, parent_hyperparam=None,
                 epsilon=None, delta=None, body=None, tail=None, expon=None):
        super(HamiltonianRule, self).__init__(
            parent_hyperparam or _default_hyperparam)
        if epsilon is not None:
            self.hyperparam.epsilon = epsilon
        if delta is not None:
            self.hyperparam.delta = delta
            
    def init_state(self, param):
        xp = backend.get_array_module(param.data)
        with cuda.get_device_from_array(param.data):
            self.state['p'] = xp.zeros_like(param.data)
        
        
    def update_core_cpu(self, param):
        grad = param.grad
        if grad is None:
            return
        hp = self.hyperparam
        p = self.state['p']
        
        p_ip1 = hp.delta * p - hp.epsilon * hp.delta * grad
        
        p_ip1var = Variable(p_ip1)
        
        sqsum = chainer.functions.sum(p_ip1var ** 2.0)
        kinetic = (1+ sqsum) ** 0.5-1
        
        kinetic.backward()
        grad_k = p_ip1var.grad
        
        p = p_ip1
        param.data += hp.epsilon * grad_k
        
     
    def update_core_gpu(self, param):
        grad = param.grad
        if grad is None:
            return
        hp = self.hyperparam
        p = self.state['p']
        
        p_ip1 = hp.delta * p - hp.epsilon * hp.delta * grad
        
        p_ip1var = Variable(p_ip1)

        sqsum = chainer.functions.sum(p_ip1var ** 2.0)
        kinetic = (1+ sqsum ) ** 0.5-1
        
        kinetic.backward()
        grad_k = p_ip1var.grad
        
        p = p_ip1
        param.data += hp.epsilon * grad_k
        

class Hamiltonian(optimizer.GradientMethod):
    def __init__(self,
                  epsilon=_default_hyperparam.epsilon,
                  delta=_default_hyperparam.delta):
        
        super(Hamiltonian, self).__init__()
        self.hyperparam.epsilon = epsilon
        self.hyperparam.delta = delta
        
    epsilon = optimizer.HyperparameterProxy('epsilon')
    delta = optimizer.HyperparameterProxy('delta')
    
    def create_update_rule(self):
        return HamiltonianRule(self.hyperparam)
        