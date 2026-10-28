from runs import Run


run = Run.load("data/test.pt")
print(run.modules["model"])


import matplotlib.pyplot as plt
plt.plot(run.hist_contig("loss"))
plt.show()

