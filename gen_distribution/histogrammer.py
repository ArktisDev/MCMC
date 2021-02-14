import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

f = open("gen_distribution/distribution.data")
lines = f.readlines()
f.close()

N = 1<<20
total_threads = 1

# preallocate the array, otherwise it is SLOW
y = [None] * total_threads * N
i = 0

for line in lines:
    y[i] = float(line)
    i = i + 1

print("Done reading file")

bins = 100

h = plt.hist(y,bins=bins)
plt.xlabel("X")
plt.ylabel("P(X)")
plt.title("M-H Method on Distribution")

plt.savefig("gen_distribution/distribution.png",dpi=400)

plt.close()

heights = h[0]
bindivs = h[1]

x = []
y = heights

for i in range(0,len(heights),1):
    x = x + [(bindivs[i] + bindivs[i + 1])/2]

# WS Distribution, mirrored
def fit_dist(x, rhoa, c, a):
    return (rhoa / (1 + np.exp((abs(x) - c) / a)))

popt, pcov = curve_fit(fit_dist, x, y, bounds=([0.,-10.,0.1], [100000., 10., 1.0]))

print(popt)

y_fit = []

for xs in x:
    y_fit = y_fit + [fit_dist(xs, *popt)]

plt.plot(x, y)
plt.plot(x, y_fit)
plt.savefig("gen_distribution/fit_distribution.png",dpi=400)