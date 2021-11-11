import torch
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")


(x_train,y_train),(x_test,y_test)=mnist.load_data()
print(y_test.shape)
x_train=x_train/255
x_test=x_test/255
x_train=np.array(x_train,dtype='f')
x_test=np.array(x_test,dtype='f')

featureTrain=torch.from_numpy(x_train)
targetTrain=torch.from_numpy(y_train).type(torch.LongTensor)

featureTest=torch.from_numpy(x_test)
targetTest=torch.from_numpy(y_test).type(torch.LongTensor)

#hyperparameters
batch_size=100
n_iters=6000 #(60.000 input // batch_size)*numberofepochs(10)
epochs=n_iters/(len(featureTrain)/batch_size)
epochs=int(epochs)

train=torch.utils.data.TensorDataset(featureTrain,targetTrain)
test=torch.utils.data.TensorDataset(featureTest,targetTest)


train_loader=DataLoader(train,batch_size=batch_size,shuffle=False)
test_loader=DataLoader(test,batch_size=batch_size,shuffle=False)

class RNNModel(nn.Module):
    def __init__(self,input_dim,hidden_dim,layer_dim,output_dim):
        super(RNNModel, self).__init__()

        self.hidden_dim=hidden_dim
        
        self.layer_dim=layer_dim
        
        self.rnn=nn.RNN(input_dim, hidden_dim,layer_dim, batch_first=True, nonlinearity='relu')
        
        self.fc=nn.Linear(hidden_dim, output_dim)
        
    def forward(self,x):
        h0=Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        
        out, hn=self.rnn(x,h0)
        out=self.fc(out[:, -1, :])
        
        return out

input_dim=28
hidden_dim=100
layer_dim=1
output_dim=10

model=RNNModel(input_dim,hidden_dim,layer_dim,output_dim)

#crossentropyloss
error=nn.CrossEntropyLoss()

#SGD optimizer
lr=0.05
optimizer=torch.optim.SGD(model.parameters(),lr=lr)

#RNN model training
seq_dim=28
count=0
loss_list=[]
iteration_list=[]
accuracy_list=[]
for epoch in range(epochs):
    for i, (images,labels) in enumerate(train_loader):
        train=Variable(images.view(-1, seq_dim, input_dim))
        print(train.shape)
        
        labels=Variable(labels)
        
        optimizer.zero_grad()
        
        outputs=model(train)
        
        loss=error(outputs,labels)
        
        loss.backward()
        
        optimizer.step()
        
        count+=1
        
        if count%600==0:
             #accuracy
            correct=0
            total=0

            for images,labels in test_loader:
                test=Variable(images.view(-1, seq_dim, input_dim))

                outputs=model(test)
                predicted=torch.max(outputs.data,1)[1]

                total+=labels.size(0)
                correct+=(predicted==labels).sum()

            accuracy=100*correct/float(total)

            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
            if count%600==0:
                print("iteration: {} Loss: {} Accuracy: {}%".format(count,loss.data,accuracy))
        
        
        



