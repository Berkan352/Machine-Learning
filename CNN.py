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
n_iters=18000 #(60.000 input // batch_size)*numberofepochs(30)
epochs=n_iters/(len(featureTrain)/batch_size)
epochs=int(epochs)

train=torch.utils.data.TensorDataset(featureTrain,targetTrain)
test=torch.utils.data.TensorDataset(featureTest,targetTest)


train_loader=DataLoader(train,batch_size=batch_size,shuffle=False)
test_loader=DataLoader(test,batch_size=batch_size,shuffle=False)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        #Convolutional layer1
        self.cnn1=nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,stride=1,padding=0)
        self.relu1=nn.ReLU()
        #maxpooling
        self.maxpool1=nn.MaxPool2d(kernel_size=2)

        #Convolutional layer1
        self.cnn2=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1,padding=0)
        self.relu2=nn.ReLU()
        #maxpooling
        self.maxpool2=nn.MaxPool2d(kernel_size=2)

        #fully connected
        self.fc1=nn.Linear(32*4*4,10)
    def forward(self,x):
        #conv1
        out=self.cnn1(x)
        out=self.relu1(out)
        out=self.maxpool1(out)
        
        #conv2
        #conv1
        out=self.cnn2(out)
        out=self.relu2(out)
        out=self.maxpool2(out)
        
        #flatten
        out=out.view(out.size(0),-1)
        
        out=self.fc1(out)
        
        return out

model=CNNModel()

#crossentropyloss
error=nn.CrossEntropyLoss()

#SGD optimizer
lr=0.02
optimizer=torch.optim.SGD(model.parameters(),lr=lr)

#CNN model training
count=0
loss_list=[]
iteration_list=[]
accuracy_list=[]
for epoch in range(epochs):
    for i, (images,labels) in enumerate(train_loader):
        train=Variable(images.view(100,1,28,28))
        labels=Variable(labels)
        
        optimizer.zero_grad()
        
        outputs=model(train)
        
        loss=error(outputs,labels)
        
        loss.backward()
        
        optimizer.step()
        
        count+=1
        
        if count%10==0:
             #accuracy
            correct=0
            total=0

            for images,labels in test_loader:
                test=Variable(images.view(100,1,28,28))

                outputs=model(test)
                predicted=torch.max(outputs.data,1)[1]

                total+=len(labels)
                correct+=(predicted==labels).sum()

            accuracy=100*correct/float(total)

            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
        if count%10==0:
            print("iteration: {} Loss: {} Accuracy: {}%".format(count,loss.data,accuracy))

plt.figure(figsize=(25,25))
plt.plot(loss_list,c="orange")
plt.xlabel("Steps",fontsize=25)
plt.ylabel("Loss",fontsize=25)
plt.show()
        
        
        



