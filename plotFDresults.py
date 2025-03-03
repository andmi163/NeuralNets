import numpy as np
import matplotlib.pyplot as plt

NepochVec = [10, 50, 100, 250, 500, 750, 1000] 

errBackProp = np.load("data/errBackProp.npy")
errFD = np.load("data/errFD.npy")
totBackTime = np.load("data/totBackTime.npy")
totFDTime = np.load("data/totFDTime.npy")

# Plot test errors against time
fig, axTime = plt.subplots()
axTime.plot(totBackTime,errBackProp,"b",
        totFDTime,errFD,"r")
axTime.legend(( "Backpropagation", "Central Difference"),fontsize=14)
axTime.set_yscale('log')
axTime.set_xscale('log')
axTime.set_xlabel("wall time (s)", fontsize = 21)
axTime.set_ylabel(r"$E_{test}$", fontsize = 21,rotation = 0)
axTime.tick_params(axis='both', which='major', labelsize=14)
axTime.grid()

# Plot test errors against Epochs
fig, ax = plt.subplots()
ax.plot(NepochVec,errBackProp,"b",
       NepochVec,errFD,"r")
ax.legend(( "Backpropagation", "Central Difference"),fontsize=14)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel("Epoch", fontsize = 21)
ax.set_ylabel(r"$E_{test}$", fontsize = 21,rotation = 0)
ax.set_xlim(NepochVec[0],NepochVec[-1])
ax.tick_params(axis='both', which='major', labelsize=14)
ax.grid()

plt.show()