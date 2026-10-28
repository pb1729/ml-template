import torch
from torch import nn

from runs import Run
from utils import must_be
from muon import MuonAsdf, param_filter_matrix, param_filter_other


class Test(nn.Module):
  def __init__(self, chan_in:int):
    super().__init__()
    self.layer1 = nn.Linear(chan_in, 20)
    self.layer2 = nn.Linear(20, 10)
  def forward(self, x):
    return self.layer2(self.layer1(x))


# It's very important to wrap all of our training code in a module name check like this.
# Otherwise it will be re-executed when we reload the run, just like our definitions.
if __name__ == "__main__":
  import argparse
  
  parser = argparse.ArgumentParser()
  parser.add_argument("--count")
  parser.add_argument("--verbose", action="store_true")
  args = parser.parse_args()
  
  run = Run()
  run.register_args(args)
  run.register_module_class(Test)

  model = run.make_module("model", "Test", 16)
  optimizer_matrix = MuonAsdf(param_filter_matrix(model.parameters()),
    lr=0.001, weight_decay=0.1)
  optimizer_other = torch.optim.AdamW(param_filter_other(model.parameters()),
    lr=0.001, weight_decay=0.1)
  
  for i in range(100):
    x = torch.randn(32, 16)
    y = model(x)
    loss = (y**2).mean()
    
    # backprop and update
    optimizer_matrix.zero_grad()
    optimizer_other.zero_grad()
    loss.backward()
    optimizer_matrix.step()
    optimizer_other.step()

    # record stats
    print(f"{run.nsteps()}\t {loss.item()}")
    run.log("loss", loss.item())
    run.step()

  run.save("data/test.pt")


