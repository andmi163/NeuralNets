import numpy as np
import matplotlib.pyplot as plt

class nn:
    def __init__(n, nNodes):
        n.nHLayer = len(nNodes)-1
        n.nNodes = nNodes
        n.w = []
        n.b = []
        n.y = []
        n.x = []
        for i in range(len(nNodes)-1):
            n.w.append(np.random.randn(nNodes[i+1],nNodes[i]))
            n.b.append(np.random.randn(nNodes[i+1],1))

    def forward(n,x):
        n.y = []
        n.x = []
        n.y.append(np.array(x, ndmin=2).T)
        for i in range(1,n.nHLayer+1):
            n.x.append(np.matmul(n.w[i-1],n.y[i-1]) + n.b[i-1])
            n.y.append(n.active(n.x[i-1]))
        return n.y[-1] 

    def active(n,x):
        return np.piecewise(x,
                            [x > 0],
                            [lambda i: 1 / (1 + np.exp(-i)), lambda i: np.exp(i) / (1 + np.exp(i))],
                            )

    
    def activeD(n,y):
        return y*(1-y)
    
    def backward(n,diffErr):
        DEDy = diffErr # n.y[-1] - n.f(n.y[0])
        DEDx = DEDy*n.activeD(n.y[-1])
        DEDw = [DEDx@n.y[-2].T]
        DEDb = [DEDx]
        for i in range(n.nHLayer-1,0,-1):
            DEDy = n.w[i].T@DEDx
            DEDx = DEDy*n.activeD(n.y[i])
            DEDw.insert(0,DEDx@n.y[i-1].T)
            DEDb.insert(0,DEDx)
        return DEDw, DEDb 
                 
    def updatePars(n,DEDw,DEDb,eps):
        for i in range(n.nHLayer-1):
            n.w[i] = n.w[i] - eps*DEDw[i]
            n.b[i] = n.b[i] - eps*DEDb[i]

# Stucture of the network
nStruct = np.array([784,500,250,100,10])

# Set number of training data, epochs and test data
Ntrain = 240000
Nbatch = 10
Nepoch = 24000
Ntest = 1000

# Set hyperparameters for FD and Gradient descent
eps = .5 # 2 # 0.01
trainMode = 2 # 1 = Update weights for every training point(on-line GD), 2 = Update weights using mean gradient from all training data (GD)

if trainMode == 1:
    loss = np.zeros(Ntrain)
elif trainMode == 2:
    loss = np.zeros(Nepoch)

# Import training and test data
TrainMat = np.load("C:/Users/mandr/Documents/PHD/NeuralNets/HandwrittenDigits/TrainDigits.npy")
TrainLabel = np.load("C:/Users/mandr/Documents/PHD/NeuralNets/HandwrittenDigits/TrainLabels.npy")
TestMat = np.load("C:/Users/mandr/Documents/PHD/NeuralNets/HandwrittenDigits/TestDigits.npy")
TestLabel = np.load("C:/Users/mandr/Documents/PHD/NeuralNets/HandwrittenDigits/TestLabels.npy")

# Contruct network
digitsN = nn(nStruct)

trainBatch = np.zeros((784,Nbatch,int(Ntrain/Nbatch)))
yDat = np.zeros((10,Nbatch,int(Ntrain/Nbatch)))
for i in range(int(Ntrain/Nbatch)):
    trainBatch[:,:,i] = TrainMat[:,i*Nbatch:(i+1)*Nbatch]
    for j in range(Nbatch):
        yDat[TrainLabel[0,i*Nbatch + j],j,i] = 1

# for i in range(Ntrain):
#     yDat[TrainLabel[0,i],i] = 1

# Training
if trainMode == 1:
    for i in range(Ntrain):

        diffErr = digitsN.forward(TrainMat[:,i]) - np.array(yDat[:,i], ndmin=2).T
        loss[i] = (0.5 * diffErr ** 2).sum().item()
        
        DEDw, DEDb = digitsN.backward(diffErr)

        digitsN.updatePars(DEDw, DEDb ,eps)

elif trainMode == 2:
    for i in range(Nepoch):
        # idBatch = np.random.randint(Ntrain,size=(Nbatch))
        # diffErr = digitsN.forward(TrainMat[:,idBatch[0]]) - np.array(yDat[:,idBatch[0]], ndmin=2).T
        diffErr = digitsN.forward(trainBatch[:,0,i]) - np.array(yDat[:,0,i], ndmin=2).T
        loss[i] += (0.5 * diffErr ** 2).sum().item()
        Rw, Rb = digitsN.backward(diffErr)
        for j in range(1,Nbatch):
            diffErr = digitsN.forward(trainBatch[:,j,i]) - np.array(yDat[:,j,i], ndmin=2).T
            loss[i] += (0.5 * diffErr ** 2).sum().item()

            DEDw, DEDb = digitsN.backward(diffErr)

            for k in range(len(nStruct)-1):
                Rw[k] += DEDw[k]
                Rb[k] += DEDb[k]

        loss[i] /= Nbatch
        for k in range(len(nStruct)-1):
            Rw[k] /= Nbatch
            Rb[k] /= Nbatch
        digitsN.updatePars(Rw,Rb,eps)

# Plot Loss
plt.yscale("log")
plt.xscale("log")
plt.plot(loss)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.show()

print("Final loss = ",loss[-1])

# Compute percentage classified correctly out of test data
totClass = 0
for i in range(Ntest):
    totClass += int(np.equal(np.argmax(digitsN.forward(TestMat[:,i])),TestLabel[:,i].item()))

percentage = (totClass/Ntest)*100
print("Classified %.0f%% of digits" % (percentage))
