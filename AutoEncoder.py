#combine stochastic/ minibatch-gradient descent for autoencoder

from AutoLib import sigmoid, randWeight, ReLU, gradSig, gradReLU, standardScaler
import numpy as np
import sys
sys.path.insert(0, "/home/tuan/Desktop/Katakana/getData")
from getData import getData, getData1
import cv2

data = getData1("/home/tuan/Desktop/Katakana/test/0")
trainX = data
trainX = trainX.T
#print trainX.shape
batch = 10
sizeInput = 50 * 50
sizeHidden = 50*50
m = data.shape[1] # size of training data
W1 = randWeight(sizeInput, sizeHidden)
b1 = randWeight(1, sizeHidden)
#print b1.shape
W2 = randWeight(sizeHidden, sizeInput)
b2 = randWeight(1, sizeInput)
#print b2.shape
decay = 3e-6
peta = 2e-3 # sparsity term
p = 0.05 #sparsity parameter
alpha = 0.1 #gradient parameter

#compute averange of training data in hidden layer
sumA2 = np.zeros((sizeHidden, 1))


for i in range(m) :
    X = np.array([trainX[:, i]])
    X = X.T
    z2 = np.dot(W1, X)+ b1
    a2 = sigmoid(z2)
    sumA2 = sumA2 + a2

print sumA2
p2 = sumA2 / (m * 1.0)
print p2


#run miniBatch
miniBatch = [trainX[:, k:k+batch] for k in range(0, m, batch)]

for i in range(len(miniBatch)) :

    #forward to compute cost
    X = miniBatch[i]
    z2 = np.dot(W1, X) + b1
    a2 = sigmoid(z2)
    z3 = np.dot(W2, a2) + b2
    a3 = sigmoid(z3)

    cost = (1.0 / (2 * X.shape[1])) * sum(sum( (a3 - X)**2 ))
    decayTerm = (decay / 2) * (sum(sum(W1**2)) + sum(b1**2) + sum(sum(W2**2)) + sum(b2**2))
    sparsityTerm = peta * sum(p * np.log2(p / p2) + (1 - p) * np.log2((1 - p) / (1 - p2)))
    J = cost + decayTerm + sparsityTerm
    print "cost : {}".format(J)

    # compute grad through backpropagation
    delta3 = (a3 - X) * gradSig(z3)
    gradSparity = peta * (-p / p2 + (1 - p) / (1 - p2))
    delta2 = (np.dot(W2.T, delta3) + gradSparity) * gradSig(z2)

    gradW1 = np.dot(delta2, X.T) + decay * W1
    gradW2 = np.dot(delta3, a2.T) + decay * W2
    gradb1 = np.array([sum(delta2.T)]).T
    gradb2 = np.array([sum(delta3.T)]).T

    #update
    W1 = W1 - alpha * gradW1
    W2 = W2 - alpha * gradW2
    b1 = b1 - alpha * gradb1
    b2 = b2 - alpha * gradb2

print W1
imgTest = cv2.imread('/home/tuan/Desktop/Katakana/test/0/image0.png', 0)
cv2.imshow("Tuan", imgTest); cv2.waitKey(0)
imgTest = imgTest.reshape(50*50, 1) * 1.0
#imgTest = standardScaler(imgTest)
img = sigmoid(np.dot(W1, imgTest) + b1)
img = sigmoid(np.dot(W2, img) + b2)
img = img.reshape(50, 50)
img = img * 255
np.savetxt('tuan.txt', img)
cv2.imshow("tuan", img); cv2.waitKey(0)
