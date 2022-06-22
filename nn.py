import math
from random import random

def Train(n):
  for i in range(0, n):
    nn.backprop(tr[math.floor(random() * len(tr))])

class TR:
  def __init__(self, inp, out):
    self.inp = inp
    self.out = out

class network:
  def __init__(self, layers):
    self.layers = layers
    self.L = len(self.layers) - 1

    self.w = [[]]
    self.b = [[]]
    self.a = []
    self.out = []

    self.lr = 0.1
    self.dCda = []

    self.initialize()
  
  def initialize(self):
    for l in range(1, self.L + 1):
      wl = []
      bl = []
      self.dCda.append([])

      for k in range(0, self.layers[l - 1]):
        self.dCda[l - 1].append(0)

      for j in range(0, self.layers[l]):
        wlj = []

        for k in range(0, self.layers[l - 1]):
          wlj.append(random() * 2 - 1)
        
        bl.append(random() * 2 - 1)
        wl.append(wlj)
      
      self.w.append(wl)
      self.b.append(bl)

  def feedforward(self, inp):
    self.a = [inp]

    for l in range(1, self.L + 1):
      aa = []

      for j in range(0, self.layers[l]):
        w = self.w[l][j]
        b = self.b[l][j]
        x = self.a[l - 1]

        z = b

        for k in range(0, self.layers[l - 1]):
          z += w[k] * x[k]
        
        a = 1 / (1 + math.exp(-z))
        aa.append(a)
      
      self.a.append(aa)
    
    self.out = self.a[self.L]
    return self.out

  def backprop(self, tr):
    self.feedforward(tr.inp)
    y = tr.out # desired out

    for i in range(0, self.L):
      l = self.L - i

      for k in range(0, self.layers[l - 1]):
        self.dCda[l - 1][k] = 0
      
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
    
  def error(self, tr):
    fw = 0

    for t in tr:
      self.feedforward(t.inp)
      for j in range(0, len(self.out)):
        fw += pow(self.out[j] - t.out[j], 2)
    
    fw /= len(tr)
    return fw

def Instructions():
  print("Train(n)")
  print("TR(inp, out)")
  print("nn: feedforward(inp (as array) ), backprop(tr (as object) ), error(tr (as array of objects) )")
  print("Start() to initialize nn and tr")

def Start():
  global nn
  global tr

  nn = network([2, 3, 3, 2])
  tr = [
    TR(
      [1, 0],
      [1, 0]
    ),
    TR(
      [0, 1],
      [1, 0]
    ),
    TR(
      [1, 1],
      [0, 1]
    ),
    TR(
      [0, 0],
      [0, 1]
    )
  ]

  # print(nn.feedforward([1, 0]))
  # Train(1000000)
  # print(nn.error(tr))
  # print(nn.feedforward([1, 0]))

Start()
Instructions()
