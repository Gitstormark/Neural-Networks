import numpy as np
import pandas as pd
from nn_general import Linear,Sigmoid,Tanh,ReLU,Softmax,CrossEntropyLoss,SGD,Model

data_train = 'train.csv'
df_train=pd.read_csv(data_train)
train_data = np.asmatrix(df_train)
y_train = train_data[:10000,0]
x_train = train_data[:10000,1:]
y_test = train_data[10000:11000,0]
x_test = train_data[10000:11000,1:]

model = Model()
model.uploadTrain(x_train,y_train)
model.add_layer(128,ReLU)
model.add_layer(10,Softmax)
model.loss_cal(CrossEntropyLoss)
model.optimize(SGD,0.01,100)

Test_loss,Test_accuracy = model.Test(x_test,y_test)
print(f"Test Loss = {Test_loss} ,Test Accuracy = {Test_accuracy}")