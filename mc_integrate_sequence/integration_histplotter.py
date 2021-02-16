import matplotlib.pyplot as plt
import numpy as np

f = open("mc_integrate.data")
lines = f.readlines()
f.close()


total_threads = 32 * 256

# preallocate the array, otherwise it is SLOW
y = [None] * total_threads
i = 0

for line in lines:
    y[i] = float(line)
    i = i + 1

print("Done reading file")

print(np.mean(y))
print(np.std(y))

bins = 100

h = plt.hist(y,bins=bins)
plt.xlabel("MC-Result")
plt.ylabel("Count")
plt.title("Monte Carlo Integration Result Distribution")

plt.savefig("mc_integrate_hist.png",dpi=400)