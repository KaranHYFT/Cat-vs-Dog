#!/usr/bin/env python
# coding: utf-8

# In[32]:


import os #directories and path
import torch #python api neural networks
from torch import nn
from torch import optim
from torchvision import datasets
from torchvision import transforms
from torchvision import models
from torchvision.models import densenet121
from collections import OrderedDict

from google.colab import drive #read data from drive
drive.mount('/content/drive', force_remount=True)


# In[33]:


#fetching current directory
print(os.getcwd())


# In[34]:


#defining normalize including mean and standard deviation
normalizeData = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# Transforming test and train data as per model specifications
TransformTraining = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       normalizeData])

TransformTesting = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      normalizeData])


# In[35]:


# loading train and test for images and running transforms
get_ipython().run_line_magic('cd', '')
get_ipython().run_line_magic('cd', '/content/drive/My Drive/Colab Notebooks/Data1/')
get_ipython().run_line_magic('cd', 'train')
train_data = datasets.ImageFolder(os.getcwd(), transform=TransformTraining)
get_ipython().run_line_magic('cd', '')
get_ipython().run_line_magic('cd', '/content/drive/My Drive/Colab Notebooks/Data1/')
get_ipython().run_line_magic('cd', 'test')
test_data = datasets.ImageFolder(os.getcwd(), transform=TransformTesting)

loadtrain = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
loadtest = torch.utils.data.DataLoader(test_data, batch_size=64)


# In[36]:


#Calling pytorch model which we will be using
modelUsed = models.densenet121(pretrained=True)
modelUsed


# In[37]:


# Freezing parameters to avoid during backpropogation
for param in modelUsed.parameters():
    param.requires_grad = False

# Module layers
cf = nn.Sequential(OrderedDict([
                          ('lin1', nn.Linear(1024, 512)),
                          ('relu1', nn.ReLU()),
                          ('lin2', nn.Linear(512,256)),
                          ('relu2', nn.ReLU()),
                          ('lin3', nn.Linear(256, 2)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
modelUsed.cf = cf


# In[38]:


# Will use GPU if available else will use CPU
deviceUsed = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelUsed = models.densenet121(pretrained=True)

# Freezing parameters to avoid during back propogation
for param in modelUsed.parameters():
    param.requires_grad = False
    
modelUsed.cf = nn.Sequential(nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.1),
                                 nn.Linear(256, 2),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

# Classifier parameters training
optimizer = optim.Adam(modelUsed.cf.parameters(), lr=0.003)
modelUsed.to(deviceUsed);


# In[39]:


#check for device/GPU availablity
torch.cuda.is_available()


# In[40]:


accuracy = []
trainloss = []
runloss = 0
test_loss = []
allsteps = []
stepscount = 0
totalepochs = 1
#print output at every 5 steps
iterateprint = 5
for epoch in range(totalepochs):
    for inputs, labels in loadtrain:
        stepscount += 1
        # will move labels and inputs to our device
        inputs, labels = inputs.to(deviceUsed), labels.to(deviceUsed)        
        optimizer.zero_grad()
        
        logps = modelUsed.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step(
        runloss += loss.item()
        
        if stepscount % iterateprint == 0:
            testrunloss = 0
            accuracy = 0
            modelUsed.eval()
            with torch.no_grad():
                for inputs, labels in loadtest:
                    inputs, labels = inputs.to(deviceUsed), labels.to(deviceUsed)
                    logps = modelUsed.forward(inputs)
                    batchtotalloss = criterion(logps, labels)
                    testrunloss += batchtotalloss.item()
                    
                    # Calculating accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
 
            # collecting each individual value based on logic and printing step wise
            trainloss.append(runloss/iterateprint)
            test_loss.append(testrunloss/len(loadtest))
            accuracy.append(accuracy/len(loadtest))
            allsteps.append(stepscount)
            print(f"Device {deviceUsed}.."
                  f"Epoch {epoch+1}/{totalepochs}.. "
                  f"Step {stepscount}.. "
                  f"Train loss: {runloss/iterateprint:.3f}.. "
                  f"Test loss: {testrunloss/len(loadtest):.3f}.. "
                  f"Test accuracy: {accuracy/len(loadtest):.3f * 100} ")
            runloss = 0
            modelUsed.train()

