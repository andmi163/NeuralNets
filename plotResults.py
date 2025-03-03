import numpy as np
import matplotlib.pyplot as plt
import os

NepochVec = [10, 50, 100, 250, 500, 750, 1000] 
NepochVec2 = [2, 4, 8, 16, 32, 64, 120, 256, 512, 1024, 2048, 4096, 8192]

# pathErrBackProp = os.path.join("data","errBackProp.npy")
# pathErrFD = os.path.join("data","errFD.npy")
# pathTotBackTime = os.path.join("data","totBackTime.npy")
# pathFDTime = os.path.join("data","totFDTime.npy")
# pathPercentage =  os.path.join("data","percentage.npy")

errBackProp = np.load('C:/Users/mandr/Documents/PHD/NeuralNets/data/errBackProp.npy')
errFD = np.load('C:/Users/mandr/Documents/PHD/NeuralNets/data/errFD.npy')
errGD = np.load('C:/Users/mandr/Documents/PHD/NeuralNets/data/errGD.npy')
errMomentum = np.load('C:/Users/mandr/Documents/PHD/NeuralNets/data/errMomentum.npy')
totBackTime = np.load('C:/Users/mandr/Documents/PHD/NeuralNets/data/totBackTime.npy')
totFDTime = np.load('C:/Users/mandr/Documents/PHD/NeuralNets/data/totFDTime.npy')
percentage = np.load('C:/Users/mandr/Documents/PHD/NeuralNets/data/percentage.npy')

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

# Plot classification results
fig, axClass = plt.subplots()
axClass.plot(np.linspace(1,100,100),percentage,"b")
axClass.set_xlabel("Epoch", fontsize = 21)
axClass.set_ylabel(r"$\%$ classified", fontsize = 21)
axClass.set_xlim(0,100)
axClass.tick_params(axis='both', which='major', labelsize=14)
axClass.grid()

# Plot momentum test results
fig, axMom = plt.subplots()
axMom.plot(NepochVec2,errGD,"b",
           NepochVec2,errMomentum,"r")
axMom.legend(( "Gradient Descent (GD)", "Momentum"),fontsize=14)
axMom.set_yscale('log')
axMom.set_xscale('log')
axMom.set_xlabel("Epoch", fontsize = 21)
axMom.set_ylabel(r"$E_{test}$", fontsize = 21,rotation = 0)
axMom.set_xlim(NepochVec[0],NepochVec[-1])
axClass.tick_params(axis='both', which='major', labelsize=18)
axMom.grid()

plt.show()