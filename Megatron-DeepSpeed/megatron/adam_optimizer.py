import torch

class CPUAdamOptimizer:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0
        
        for param in parameters:
            self.m[param] = torch.zeros_like(param)
            self.v[param] = torch.zeros_like(param)

    def zero_grad(self):
        for param in self.m:
            self.m[param].zero_()
        for param in self.v:
            self.v[param].zero_()

    def step(self, parameters, grads):
        self.t += 1
        
        for param in parameters:
            grad = grads[param]
            self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * grad
            self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * (grad ** 2)
            
            m_hat = self.m[param] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param] / (1 - self.beta2 ** self.t)
            
            param -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
