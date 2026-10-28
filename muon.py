import torch
import torch.nn as nn


EPS = 1e-8


def param_filter_matrix(params):
  for param in params:
    if len(param.shape) >= 2:
      yield param

def param_filter_other(params):
  for param in params:
    if len(param.shape) < 2:
      yield param


def ns_iter(A, a1=1.5, a3=-0.5):
  # for fast convergence, use 
  A_cube = torch.einsum("ij..., kj..., kl... -> il...", A, A, A)
  return a1*A + a3*A_cube


class MuonAsdf(torch.optim.Optimizer):
  def __init__(self, params, lr:float=1e-3, momentum:float=0.9, weight_decay=0.1):
    super().__init__(params, defaults={"lr": lr, "weight_decay": weight_decay})
    self.momentum = momentum
    self.state = {}
    for group in self.param_groups:
      for param in group["params"]:
        self.state[param] = {"vel": torch.zeros_like(param.data)}
  @torch.no_grad() # we don't take gradients inside of optimizer
  def step(self):
    for group in self.param_groups:
      for param in group["params"]:
        vel = self.state[param]["vel"]
        vel.mul_(self.momentum)
        vel.add_(param.grad.data)
        # first normalize by the frobenius norm
        n, m, *rest = vel.shape
        norm_f = torch.sqrt(torch.square(vel).sum((0, 1)))
        A = vel/(EPS + norm_f)
        # faster-converging initial iteration
        for i in range(4):
          A = ns_iter(A, a1=2.0, a3=-1.0)
        # fine-tuning iteration
        for i in range(1):
          A = ns_iter(A)
        # update the parameters
        param.mul_(1. - group["lr"]*group["weight_decay"])
        param.add_(-group["lr"]*(max(n, m)**0.5)*A)




