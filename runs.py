import sys
import inspect
import re
import importlib
from dataclasses import dataclass

import torch


# constants:
DEFAULT_EXCLUDE = [r"(/site-packages/)|(/lib/python3\.[0-9]+/)"]
# globals:
G_IN_EXEC_SOURCES:bool = False


def is_excluded(patterns:list[str], path:str) -> bool:
  for pattern in patterns:
    if re.search(pattern, path) is not None: return True
  return False

def get_source(exclude:list[str]=DEFAULT_EXCLUDE) -> dict[str, str]:
  ans = {}
  for key in sys.modules: # we expect the ordering here should match exec order
    try:
      path = inspect.getsourcefile(sys.modules[key])
    except TypeError: continue # skip over built-in modules
    if path is None: continue  # skip over modules with no path
    if not is_excluded(exclude, path):
      with open(path, "r") as f:
        source = f.read()
      ans[key] = source
  return ans


def string_module(name:str, source:str):
  spec = importlib.util.spec_from_loader(name, loader=None)
  module = importlib.util.module_from_spec(spec)
  exec(source, module.__dict__) # DANGER! Must trust files we load...
  return module

def exec_sources(sources:dict[str, str]):
  global G_IN_EXEC_SOURCES
  G_IN_EXEC_SOURCES = True
  for module_name in sources:
    source = sources[module_name]
    if module_name == "__main__": # need to replace this guy...
      module_name = "original_script__main__"
    sys.modules[module_name] = string_module(module_name, source)
  G_IN_EXEC_SOURCES = False
  return sys.modules["original_script__main__"] # provide access to the namespace of the original script


class Run:
  def __init__(self, empty:bool=False, exclude:list[str]=DEFAULT_EXCLUDE):
    self._check_not_executing_sources()
    self.module_classes = {} # mapping from class names to that class
    self.module_blueprints = {} # mapping from module name to (class_name, args, kwargs)
    self.modules = {} # mapping from module name to module
    self.history = []
    self.current_datapoints = None
    if empty:
      self.sources = None
    else:
      self.sources = get_source(exclude)
  def _check_not_executing_sources(self):
    global G_IN_EXEC_SOURCES
    if G_IN_EXEC_SOURCES:
      raise RuntimeError(
        "Dis-allowed operation called while executing sources during re-loading.\n"
        "Reminder: Everything in your scripts will be re-executed, "
        "unless it's wrapped in if __name__ == \"__main__\".")
  def register_module_class(self, cls):
    """ Warning: You *must* keep the class name consistent across files here. """
    self.module_classes[cls.__name__] = cls
  def make_module(self, name:str, class_name:str, *args, **kwargs) -> torch.nn.Module:
    """ Warning: It's expected args and kwargs will be nicely serializable! """
    self.module_blueprints[name] = (class_name, args, kwargs)
    self.modules[name] = self.module_classes[class_name](*args, **kwargs)
    return self.modules[name]
  @staticmethod
  def from_dict(data:dict):
    ans = Run(empty=True)
    # recover and execute all sources
    sources = data["sources"]
    script_module = exec_sources(sources)
    # fill empty fields
    ans.sources = sources
    ans.module_classes = {
      class_name: getattr(script_module, class_name)
      for class_name in data["module_classes"]
    }
    ans.module_blueprints = data["module_blueprints"]
    ans.modules = {}
    for name in data["modules"]:
      class_name, args, kwargs = data["module_blueprints"][name]
      module = ans.make_module(name, class_name, *args, **kwargs)
      module.load_state_dict(data["modules"][name]) # we have to load the saved state too
      ans.modules[name] = module
    ans.history = data["history"]
    return ans
  def to_dict(self) -> dict:
    self._check_not_executing_sources()
    assert self.current_datapoints is None, "must complete the step before exporting to dict"
    return {
      "sources": self.sources,
      "module_classes": {
        class_name: None
        for class_name in self.module_classes
      },
      "module_blueprints": self.module_blueprints,
      "modules": {
        module_name: self.modules[module_name].state_dict()
        for module_name in self.modules
      },
      "history": self.history,
    }
  def save(self, path):
    return torch.save(self.to_dict(), path)
  @staticmethod
  def load(path):
    return Run.from_dict(torch.load(path))
  def step(self):
    if self.current_datapoints is None:
      self.history.append({})
    else:
      self.history.append(self.current_datapoints)
    self.current_datapoints = None
  def log(self, key, val):
    if self.current_datapoints is None:
      self.current_datapoints = {}
    assert key not in self.current_datapoints
    self.current_datapoints[key] = val
  def get_module(self, name:str):
    return self.modules[name]
  def hist_contig(self, key:str):
    return [row[key] for row in self.history]
  def hist_scatter(self, key:str):
    x, y = [], []
    for i, row in enumerate(self.history):
      if key in row:
        x.append(i)
        y.append(row[key])
    return x, y
  def nsteps(self) -> int:
    return len(self.history)








