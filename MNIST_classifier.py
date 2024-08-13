import numpy as np
import pandas as pd
from nn_general import Linear,Sigmoid,Tanh,ReLU,Softmax,CrossEntropyLoss,SGD,Model

data_train = 'train.csv'
df_train=pd.read_csv(data_train)
train_data = np.asmatrix(df_train)
df_y = df_train["label"]
df_one_hot=pd.get_dummies(df_y,columns = ["label"],dtype = int)
y_train = np.asmatrix(df_one_hot)
y_train = y_train[:10000,:]
x_train = train_data[:10000,1:]
y_test = y_train[10000:11000,:]
x_test = train_data[10000:11000,1:]

model = Model()
model.uploadTrain(x_train,y_train)
model.add_layer(128,ReLU)
model.add_layer(10,Sigmoid)
model.loss_cal(CrossEntropyLoss)
model.optimize(SGD,0.01,100)

Test_loss,Test_accuracy = model.Test(x_test,y_test)
print(f"Test Loss = {Test_loss} ,Test Accuracy = {Test_accuracy}")