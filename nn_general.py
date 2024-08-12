import numpy as np

class Linear:
  def front(self,X,W,b):
    Z = X@W + b
    return Z

class Sigmoid:
  def front(self,Z):
    return 1/(1+np.exp(-Z))

  def back(self,Z):
    S = self.front(Z)
    S_dash = Z*(1-Z)
    return S_dash

class Tanh:
  def front(self,Z):
    return np.tanh(Z)

  def back(self,Z):
    S = self.forward(Z)
    S_dash = 1-(S**2)
    return S_dash

class ReLU:
  def front(self,Z):
    return np.maximum(0,Z)

  def back(self,Z):
    S_dash = (Z>0)*1
    return S_dash

class Softmax:
  def forward(self,Z):
    S = np.exp(Z)
    # max = np.max(S,axis=1)
    # np.reshape(max,(S.shape[0],1))
    # S = S/max
    S = S/np.max(S,axis=1,keepdims=True)
    # Sum = np.sum(S,axis=1)
    # np.reshape(Sum,(S.shape[0],1))
    # S=S/Sum
    S = S/np.sum(S,axis=1,keepdims=True)
    return S

  def backward(self):
    pass

class CrossEntropyLoss:
  def front(self,Y,S):
    #Y are true labels,S is matrix after applying activation func.
    L = -np.sum(Y*np.logS,axis=1).mean()

  def back(self,X,Y,S,S_dash):
    dL_W = -(X.T)@(Y*S_dash/S)*(1/X.shape[0])
    dL_b = -np.sum((Y*S_dash)/S,axis=0)*(1/X.shape[0])
    return dL_W,dL_b

class SGD:
  def __init__(self,learning_rate,epochs):
    self.lr = learning_rate
    self.epochs = epochs

  def update(self,W,dL_W,b,dL_b):
    for i in range(self.epochs):
      W = W - self.lr*dL_W
      b = b - self.lr*dL_b
    return W,b

class Model:
  def __init__(self):
    self.layers = []
    self.super_W = []
    self.super_b = []
    self.act_funcs = []
    self.grads_W = []
    self.grads_b = []
    
  def uploadTrain(self,x_train,y_train):
    self.X = x_train
    self.Y = y_train
    self.layers.append(self.X)


  def add_layer(self,added_layer_size,act_func):
    self.in_layer_size = (self.layers[-1]).shape[1]
    self.out_layer_size = added_layer_size
    W,b = self.generate_W_b()
    self.super_W.append(W)
    self.super_b.append(b)
    Z = Linear.front(X=self.layers[-1],W=W,b=b)
    self.act_funcs.append(act_func)
    S = {act_func}.front(Z)
    self.layers.append(S)

  def generate_W_b(self):
    W = np.random.rand(self.in_layer_size,self.out_layer_size)
    b = np.random.rand(1,self.out_layer_rize)
    return W,b

  def loss_cal(self,loss):
    for i in range(len(self.super_W)):
      X = self.layers[i]
      S = self.layers[i+1]
      Y = self.layers[i+1]
      W = self.super_W[i]
      b = self.super_b[i]
      self.loss = loss
      S_dash={self.act_funcs[i]}.back(Z=Linear.front(X,W,b))
      if i==(len(self.super_W)-1):
        Y = self.Y
      dW,db = {loss}.back(X,Y,S,S_dash)
      self.grads_W.append(dW)
      self.grads_b.append(db)

  def optimize(self,optimizer,learning_rate,epochs):
    for i in range(len(self.super_W)):
      W,dW,b,db = self.super_W[i],self.grads_W[i],self.super_b[i],self.grads_b[i]
      self.super_W[i],self.super_b[i] = {optimizer}(learning_rate,epochs).update(W,dW,b,db)
      
  def Test(self,x_test,y_test):
    self.X_test = x_test
    self.Y_test = y_test
    self.Y_pred = self.X_test
    for i in range(len(self.super_W)):
      self.Y_pred = self.act_funcs[i].front(Z=Linear.front(self.Y_pred,self.super_W[i],self.super_b[i]))
    self.predict()
    self.Test_loss = {self.loss}.front(Y=self.Y_test,S=self.Y_pred)
    self.Test_accuracy = (np.min((self.Y_true==self.Y_pred)*1,axis=1,keepdims=True).mean())*100
    return self.Test_loss,self.Test_accuracy

  def predict(self):
    self.Y_pred = (self.Y_pred == np.max(self.Y_pred,axis=1,keepdims=True))*1

  