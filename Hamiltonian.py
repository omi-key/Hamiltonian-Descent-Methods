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
_default_hyperparam.epsilon = 1
_default_hyperparam.delta = 0.6


class HamiltonianRule(optimizer.UpdateRule):
    
    """ Update rule of Hamiltonian Gradient Descent.
    
    sample ver.
    """
    
    _kernel = None
    _kernel_2 = None
    
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
        
        p = hp.delta * p - hp.epsilon * hp.delta * grad
        sqsum = np.vdot(p,p)
        
        param.data += hp.epsilon / (1.0 + sqsum) * p

    def update_core_gpu(self, param):
        grad = param.grad
        if grad is None:
            return
        hp = self.hyperparam
        p = self.state['p']
        
        if HamiltonianRule._kernel is None:
            HamiltonianRule._kernel = cuda.elementwise(
                'T delta, T epsilon, T grad', 'T p',
                'p *= delta; p -= epsilon * delta * grad;',
                'Hamiltonian_p')
        HamiltonianRule._kernel(hp.delta, hp.epsilon, grad, p)

        sqsum = cuda.cupy.vdot(p,p)
        if HamiltonianRule._kernel_2 is None:
            HamiltonianRule._kernel_2 = cuda.elementwise(
                'T epsilon, T p, T sqsum', 'T param', 
                'param += epsilon * p / (1.0 + sqsum)', 'Hamiltonian_q')
        
        HamiltonianRule._kernel_2(hp.epsilon, p, sqsum, param.data)

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
        