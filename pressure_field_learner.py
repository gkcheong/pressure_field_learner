import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class OmegaData (Dataset):
    def __init__(self, npdata):
        #npdata = np.loadtxt('clean_data',dtype = [('composition','f4'),('pressure','f4')])
        #npdata = np.loadtxt('clean_data')
        #npdata = npdata.reshape([-1,4096,2])
        #print(type(npdata))
        #print(type(npdata[0]))
        #print(npdata[0])
        #print(npdata.shape)
        self.data = torch.from_numpy(npdata[10:,:,:]).type(dtype=torch.float)
        print(self.data.size())
    
    def __len__(self):
        return self.data.size()[0]
    
    def __getitem__(self, idx):
        return self.data[idx]

class NeuralNet (nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        #self.fc3 = nn.Linear(4096, 4096)

    #I could name the input `input` i.e forward(self,x)
    #But I can reuse this value later instead of copying it
    #is return optional?
    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x
  
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    npdata = np.loadtxt('clean_data')
    npdata = npdata.reshape([-1,4096,2])
    arraySize = npdata.shape[0]
    print('data length is %i' % arraySize)
    inputs = OmegaData(npdata)
    validation = torch.from_numpy(npdata[:10,:,:]).type(dtype=torch.float)
    print('validation is %s' % (validation.size(),))
    
    fcnet = NeuralNet().to(device)
    criterion = nn.MSELoss() #reduction is mean over all batch
    optimizer = optim.SGD(fcnet.parameters(), lr = 0.01, momentum = 0.9)
    #add optimizer move container in cuda before this

    dataloader = DataLoader(inputs, batch_size = 5, shuffle=True, num_workers = 4)
    #can store every 100 iteration's loss value
    #i.e iteration 1/2450) loss : something
    #i.e Epoch 1/5 train_acc: something val_acc: something
    #overfit a small batch of data (2 layers to overfit 50 data)
    # 0.32 loss over 100 epoch, 50 image, batch_size = 25, lr = 0.01 2FC layers
    # 0.418 loss over 100 epoch 50 image, batch_size = 25, lr = 0.01 3FC layers
    # 0.00056591781321913 over 200 epoch 10 image, batch_size = 5, lr = 0.01 2 FC layers
    # some 1.644 loss over 200 epoch all data, batch_size =5, lr = 0.01 2 FC layers training loss is 0.01
    # approximately 150 epoch is enough
    # need more params?
    maxEpoch = 200
    lossList = []
    for iEpoch in range(maxEpoch):
        for iBatch, sampleFromBatch in enumerate(dataloader):
            #there are batch_size amount of them in sampleFrombatch
            #print(len(sampleFromBatch))
            #print(sampleFromBatch.size())
            #print(sampleFromBatch.storage_type())
            inputs = sampleFromBatch[:,:,0].to(device)
            labels = sampleFromBatch[:,:,1].to(device)
            #print(inputs.storage_type())
            optimizer.zero_grad()
            
            outputs = fcnet(inputs)
            loss = criterion(outputs, labels)
            lossList.append(loss)
            loss.backward()
            optimizer.step()
            #quit()
            #print('This is batch %i with loss %f' % (iBatch, loss))
            #quit()
        #print some information on loss
        
        #test
        with torch.no_grad():
            inputs = validation[:,:,0].to(device)
            labels = validation[:,:,1].to(device)
            outputs = fcnet(inputs)
            loss = criterion(outputs, labels)
            print('This is epoch %i with loss %f' % (iEpoch, loss))
            #print(type(outputs))
            #print(outputs.size)
            if iEpoch == maxEpoch - 1: #print final outputs
                print(outputs.data.size())
                listOutput = outputs.data.tolist()
                with open('final_output', 'w') as outputFile:
                    for idx in range(10):
                        for idy in range(4096):
                            outputFile.write(str(listOutput[idx][idy]) + '\n')
                        outputFile.write('\n')
    #save loss to file
    with open('helpme','w') as f:
        for loss in lossList:
            f.write(str(loss.item()) + '\n')