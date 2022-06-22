import math
from random import random

# train the network with all the training examples n times
def Train(n):
  for i in range(0, n):
    nn.backprop(tr[math.floor(random() * len(tr))])

# training example with input and output as arrays
class TR:
  def __init__(self, inp, out):
    self.inp = inp
    self.out = out

# neural network
class Network:
  def __init__(self, layers):
    self.layers = layers # array with number of neurons for each layer
    self.L = len(self.layers) - 1 # index of last layer

    self.w = [[]] # all the weights
    self.b = [[]] # all the biases
    self.a = [] # all the neuron activations
    self.out = [] # output layer

    self.lr = 0.1 # learning rate
    self.dCda = [] # derivatives Loss(Activation)

    self.initialize() # fill w, b and dCda with random values
  
  def initialize(self):
    # starting at index 1, since index 0 doesn't need w and b
    # going from 1 to L
    for l in range(1, self.L + 1):
      # create empty arrays for the information for each layer
      wl = []
      bl = []
      self.dCda.append([])

      # filling dCda with 0 for the layers 0 to L-1
      for k in range(0, self.layers[l - 1]):
        self.dCda[l - 1].append(0)

      # going through number of neurons in the l'th layer
      for j in range(0, self.layers[l]):
        wlj = []

        # filling random w for that neuron, based on number of neurons in l-1
        for k in range(0, self.layers[l - 1]):
          wlj.append(random() * 2 - 1)
        
        # set random bias for that neuron
        bl.append(random() * 2 - 1)
        wl.append(wlj)
      
      self.w.append(wl)
      self.b.append(bl)

  # feedforward, takes input as array
  def feedforward(self, inp):
    self.a = [inp] # setting the first layer activations to the input

    # looping thorugh layer index 1 to L
    for l in range(1, self.L + 1):
      aa = []

      # looping through each neuron in l'th layer
      for j in range(0, self.layers[l]):
        w = self.w[l][j] # weights from layer l-1 to this neuron (array)
        b = self.b[l][j] # bias of this neuron (number)
        x = self.a[l - 1] # activations of layer l-1 (array)

        z = b # z = ... + b

        for k in range(0, self.layers[l - 1]):
          z += w[k] * x[k] # z = w*x + w*x + ... + w*x + b
        
        # a = sigmoid(z)
        a = 1 / (1 + math.exp(-z))
        aa.append(a) # push a in activations of current l'th layer
      
      self.a.append(aa) # push activations of l'th layer into all activations of nn
    
    self.out = self.a[self.L] # output is last layer of activations
    return self.out

  # backpropagation, takes single training example with input and output
  def backprop(self, tr):
    self.feedforward(tr.inp) # fill the activations by passing input through the network
    y = tr.out # desired output

    for i in range(0, self.L):
      # l goes backwards from index L to 1
      l = self.L - i

      # set dCda's of layer l-1 to 0
      for k in range(0, self.layers[l - 1]):
        self.dCda[l - 1][k] = 0
      
      # calculate dCdw, dCdb and dCda(l-1)
      for j in range(0, self.layers[l]):
        alj = self.a[l][j]
        dCdz = alj * (1 - alj)
        if l == self.L:
          dCdz *= alj - y[j]
        else:
          dCdz *= self.dCda[l][j]
        
        self.b[l][j] -= dCdz * self.lr

        for k in range(0, self.layers[l - 1]):
          self.w[l][j][k] -= dCdz * self.a[l - 1][k] * self.lr
          if l > 1:
            self.dCda[l - 1][k] += dCdz * self.w[l][j][k]
    
  # get error (or loss) of network, takes ALL training examples
  def error(self, tr):
    c = 0 # start error at 0

    # loop thorugh every training example
    for t in tr:
      # send input into network to get output activations
      self.feedforward(t.inp)
      # loop through each output activation
      for j in range(0, len(self.out)):
        # add (difference between desired output and output)squared
        c += pow(self.out[j] - t.out[j], 2)
    
    # divide by number of training examples to get average error per training example
    c /= len(tr)
    
    return c

# create neural network and training examples
def Start():
  global nn # neural network
  global tr # training examples

  nn = Network([2, 3, 3, 2]) # layers: 2 inputs, 3 hidden, 3 hidden, 2 outputs
  
  # four examples with each 2 inputs and 2 output (must match number of inputs & outputs of network)
  # here, the network will be trained to detect, if two numbers are the same
  tr = [
    # same
    TR(
      [1, 1],
      [1, 0]
    ),
    # same
    TR(
      [0, 0],
      [1, 0]
    ),
    # different
    TR(
      [1, 0],
      [0, 1]
    ),
    # different
    TR(
      [0, 1],
      [0, 1]
    )
  ]

  print(tr[0]) # just out of curiosity: what does the network output without being trained?
  print(nn.error(tr)) # and how terrible does it perform?
  Train(10 ** 5) # now train the network that many times
  print(nn.error(tr)) # get the error after training is done
  print(tr[0]) # now feedforward the same input after the network is trained

# start everything
Start()
