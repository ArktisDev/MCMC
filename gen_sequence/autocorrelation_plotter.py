import matplotlib.pyplot as plt
import numpy as np

f = open("gen_sequence/r1_vs_sigma_step.data")
lines = f.readlines()
f.close()

x = []
y = []

for line in lines:
    splits = line.split(",")
    s = splits[0]
    r = splits[1]
    x = x + [float(s)]
    y = y + [float(r)]

plt.close()

plt.plot(x,y)
plt.xlabel("Sigma_step")
plt.ylabel("Autocorrelation")
plt.title("M-H Method on Distribution")

plt.savefig("gen_sequence/auto_corr_fig.png",dpi=400)