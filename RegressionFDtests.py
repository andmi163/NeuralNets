import numpy as np
import matplotlib.pyplot as plt
import time

class nn:
    def __init__(n, nNodes, aFunc, f):
        n.nHLayer = len(nNodes)-1
        n.nNodes = nNodes
        n.aFunc = aFunc
        n.w = []
        n.b = []
        n.y = []
        n.x = []
        n.f = f
        for i in range(len(nNodes)-1):
            n.w.append(np.random.randn(nNodes[i+1],nNodes[i]))
            n.b.append(np.random.randn(nNodes[i+1],1))

    def forward(n,x):
        n.y = []
        n.x = []
        n.y.append(np.array([[x]]))
        for i in range(1,n.nHLayer):
            n.x.append(np.matmul(n.w[i-1],n.y[i-1]) + n.b[i-1])
            n.y.append(n.active(n.x[i-1]))
        n.y.append(np.matmul(n.w[n.nHLayer-1],n.y[n.nHLayer-1]) + n.b[n.nHLayer-1])
        n.x.append(n.y[-1])
        return n.y[-1] 

    def active(n,x):
        if n.aFunc == "sigmoid":
            return 1/(1+np.exp(-x))
        elif n.aFunc == "sine":
            return np.sin(x)+x
        elif n.aFunc == "exp":
            return np.exp(x*1.j)
        else:
            print("Activation function not implemented yet")
            return -1 
    
    def activeD(n,x):
        if n.aFunc == "sigmoid":
            return np.exp(-x)/((1+np.exp(-x))**2)
        elif n.aFunc == "sine":
            return np.cos(x)+1
        elif n.aFunc == "exp":
            return 1.j*np.exp(x*1.j)
        else:
            print("Activation function derivative not implemented yet")
            return -1 
    
    def backward(n):
        DEDy = n.y[-1] - n.f(n.y[0])
        DEDx = DEDy
        DEDw = [DEDx@n.y[-2].T]
        DEDb = [DEDx]
        for i in range(n.nHLayer-1,0,-1):
            DEDy = n.w[i].T@DEDx
            DEDx = DEDy*n.activeD(n.x[i-1])
            DEDw.insert(0,DEDx@n.y[i-1].T)
            DEDb.insert(0,DEDx)
        return DEDw, DEDb 
                 
    def updatePars(n,DEDw,DEDb,eps):
        Dw = []; Db = []
        for i in range(n.nHLayer-1):
            Dw.append(-eps*DEDw[i])
            Db.append(-eps*DEDb[i])
            n.w[i] = n.w[i] + Dw[i]
            n.b[i] = n.b[i] + Db[i]
        return Dw, Db
             
    def momentumPars(n,DEDw,DEDb,eps,alpha,DwOld,DbOld):
        Dw = []; Db = []
        for i in range(n.nHLayer-1):
            Dw.append(-eps*DEDw[i])
            Db.append(-eps*DEDb[i])
            n.w[i] = n.w[i] - Dw[i] + alpha*DwOld[i]
            n.b[i] = n.b[i] - Db[i] + alpha*DbOld[i]
        return Dw, Db

    def backwardFD(n,dx):
        oldW = []
        oldB = []
        DEDw = []
        DEDb = []
        E = (0.5 * (n.y[-1] - n.f(n.y[0]))**2).item()
        for l in range(n.nHLayer):
            oldW.append(n.w[l].copy())
            oldB.append(n.b[l].copy())
            DEDw.append(np.zeros((n.nNodes[l+1],n.nNodes[l])))
            DEDb.append(np.zeros((n.nNodes[l+1],1)))
            for i in range(n.nNodes[l+1]):

                n.b[l][i,0] += dx
                n.forward(n.y[0].item())
                Eplus = (0.5 * (n.y[-1] - n.f(n.y[0]))**2).item()
                n.b[l][i,0] = oldB[l][i,0]

                n.b[l][i,0] -= dx
                n.forward(n.y[0].item())
                Eminus = (0.5 * (n.y[-1] - n.f(n.y[0]))**2).item()
                n.b[l][i,0] = oldB[l][i,0]

                DEDb[l][i,0] = (Eplus-Eminus)/(2*dx)
                for j in range(n.nNodes[l]):

                    n.w[l][i,j] += dx
                    n.forward(n.y[0].item())
                    Eplus = (0.5 * (n.y[-1] - n.f(n.y[0]))**2).item()
                    n.w[l][i,j] = oldW[l][i,j]
                    
                    n.w[l][i,j] -= dx
                    n.forward(n.y[0].item())
                    Eminus = (0.5 * (n.y[-1] - n.f(n.y[0]))**2).item()
                    n.w[l][i,j] = oldW[l][i,j]
                    
                    DEDw[l][i,j] = (Eplus-Eminus)/(2*dx)
        return DEDw, DEDb 

# Stucture of the network
nStruct = np.array([1,100,1])

NepochVec = [10, 25, 50, 75, 100, 125, 150, 175, 200] #, 1000, 2500, 5000, 7500, 10000]

M = len(NepochVec)
errBackProp = np.zeros(M)
errFD = np.zeros(M)
totBackTime = np.zeros(M)
totFDTime = np.zeros(M)

for e in range(M):
    # Set number of training data, epochs and test data
    Ntrain = 100
    Nepoch = NepochVec[e] #5000
    Ntest = 100

    # Set hyperparameters for FD and Gradient descent
    dx = 0.001
    eps = 1 # 2 # 0.01

    loss = np.zeros(Nepoch)

    trainDat = 2 * (np.pi) * (np.random.rand(Ntrain))
    yDat = np.sin(trainDat)

    # Scale the data
    yMean = np.mean(yDat)
    yStd = np.std(yDat)
    xMean = np.mean(trainDat)
    xStd = np.std(trainDat)

    trainDat2 = (trainDat - np.mean(trainDat)) / np.std(trainDat)
    yDat2 =(yDat - np.mean(yDat)) / np.std(yDat)

    sinFunc = lambda x:(np.sin((x*xStd)+xMean) - yMean)/yStd

    # Contruct network
    sinN = nn(nStruct,"sigmoid",sinFunc)

    print("Doing backprop for Nepoch = ", Nepoch)

    # Training
    sEpochBackprop = time.time()
    for i in range(Nepoch):
        diffErr = sinN.forward(trainDat2[0]) - yDat2[0]
        loss[i] += (0.5 * diffErr ** 2).item()
        Rw, Rb = sinN.backward()
        for j in range(1,Ntrain):
            diffErr = sinN.forward(trainDat2[j]) - yDat2[j]
            loss[i] += (0.5 * diffErr ** 2).item()
            DEDw, DEDb = sinN.backward()

            for k in range(len(nStruct)-1):
                Rw[k] += DEDw[k]
                Rb[k] += DEDb[k]

        loss[i] /= Ntrain
        for k in range(len(nStruct)-1):
            Rw[k] /= Ntrain
            Rb[k] /= Ntrain
        sinN.updatePars(Rw,Rb,eps)    
    eEpochBackprop = time.time()
    totBackTime[e] = eEpochBackprop - sEpochBackprop

    testDat = np.linspace(0, 2*np.pi, num=Ntest)
    trueSol = []
    netSol = []

    for i in range(Ntest):
        netSol.append(sinN.forward((testDat[i]  -np.mean(trainDat)) / np.std(trainDat) ).item()  * np.std(yDat) +  np.mean(yDat))
        trueSol.append(np.sin(testDat[i]))
        errBackProp[e] += (netSol[i] - trueSol[i])**2

    errBackProp[e] /= Ntest

    # Contruct network again
    sinN = nn(nStruct,"sigmoid",sinFunc)

    print("Doing FD for Nepoch = ", Nepoch)

    # Training
    sEpochFDprop = time.time()
    for i in range(Nepoch):
        diffErr = sinN.forward(trainDat2[0]) - yDat2[0]
        loss[i] += (0.5 * diffErr ** 2).item()
        Rw, Rb = sinN.backwardFD(dx)
        for j in range(1,Ntrain):
            diffErr = sinN.forward(trainDat2[j]) - yDat2[j]
            loss[i] += (0.5 * diffErr ** 2).item()
            DEDw, DEDb = sinN.backwardFD(dx)

            for k in range(len(nStruct)-1):
                Rw[k] += DEDw[k]
                Rb[k] += DEDb[k]

        loss[i] /= Ntrain
        for k in range(len(nStruct)-1):
            Rw[k] /= Ntrain
            Rb[k] /= Ntrain
        sinN.updatePars(Rw,Rb,eps)
    eEpochFDprop = time.time()
    totFDTime[e] = eEpochFDprop - sEpochFDprop

    testDat = np.linspace(0, 2*np.pi, num=Ntest)
    trueSol = []
    netSol = []

    for i in range(Ntest):
        netSol.append(sinN.forward((testDat[i]  -np.mean(trainDat)) / np.std(trainDat) ).item()  * np.std(yDat) +  np.mean(yDat))
        trueSol.append(np.sin(testDat[i]))
        errFD[e] += (netSol[i] - trueSol[i])**2

    errFD[e] /= Ntest

# Plot test errors against Epochs
# fig, ax = plt.subplots()
# ax.plot(NepochVec,errBackProp,"b",
#        NepochVec,errFD,"r")
# ax.legend(( "Backpropagation", "Central Difference"),fontsize=14)
# ax.set_yscale('log')
# ax.set_xscale('log')
# ax.set_xlabel("Epoch", fontsize = 21)
# ax.set_ylabel(r"$E_{test}$", fontsize = 21,rotation = 0)
# ax.set_xlim(NepochVec[0],NepochVec[-1])
# ax.tick_params(axis='both', which='major', labelsize=14)
# ax.grid()

with open('errBackProp.npy', 'wb') as f:
    np.save(f, errBackProp)
with open('errFD.npy', 'wb') as f:
    np.save(f, errFD)
with open('totBackTime.npy', 'wb') as f:
    np.save(f, totBackTime)
with open('totFDTime.npy', 'wb') as f:
    np.save(f, totFDTime)

# Plot test errors against time
# fig, ax = plt.subplots()
# ax.plot(totBackTime,errBackProp,"b",
#        totFDTime,errFD,"r")
# ax.legend(( "Backpropagation", "Central Difference"),fontsize=14)
# ax.set_yscale('log')
# ax.set_xscale('log')
# ax.set_xlabel("wall time (s)", fontsize = 21)
# ax.set_ylabel(r"$E_{test}$", fontsize = 21,rotation = 0)
# ax.tick_params(axis='both', which='major', labelsize=14)
# ax.grid()