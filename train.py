from nplUtil import tokenize,stemm,bag
import json
import torch
import numpy as np
import random
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from neural_n import NeuralNet

ignored_char = ["?","!",".",","];
trainX = []
trainY = []
tags = []
wordlist = []
xy = []


class File():
    path = "./dataset/model.json"

models = File()

with open(models.path,"r") as model:
    models = json.load(model)

with open("config.json","r") as configs:
    config = json.load(configs)
    
for data in models:
    tag = str(data)
    tags.append(tag)
    for pair in models[str(data)]:
        words = tokenize(pair['query'])
        words = [stemm(char) for char in words if char not in ignored_char]
        wordlist.extend(words)
        xy.append((words,tag))
        
for (word,tag) in xy:
    bags = bag(wordlist,word)
    trainX.append(bags)
    label = tags.index(tag)
    trainY.append(label)
    
traniX = np.array(trainX)
trainY = np.array(trainY)
    
class ChatDataSet(Dataset):
    def __init__(self):
        self.n_samples = len(trainX)
        self.x_data = trainX
        self.y_data = trainY
    
    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    
    def __len__(self):
        return self.n_samples
    
class Config():
    learning_rate = config['learning_rate']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    hidden_size = 8
    output_size = len(tags)
    input_size = len(trainX[0])

trainConfig = Config()

dataSet = ChatDataSet()
trainLoader = DataLoader(dataset=dataSet,batch_size=trainConfig.batch_size,shuffle=True,num_workers=0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size=trainConfig.input_size,hidden_size=trainConfig.hidden_size,output_size=trainConfig.output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=trainConfig.learning_rate)

counter = 0
progress = 0

for epoch in range(trainConfig.num_epochs):
    counter += 1
    if counter/trainConfig.num_epochs == 0.01 :
        counter = 0
        progress += 1
        print("Progress : ",progress,"%")
    for (word,label) in trainLoader:
        word = word.to(device)
        label = label.to(dtype=torch.long).to(device)
        
        output = model(word)
        loss = criterion(output,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0 :
        print(f'epoch {epoch+1}/{trainConfig.num_epochs},loss={loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')

data = {
    "model_state" : model.state_dict(),
    "input_size" : trainConfig.input_size,
    "output_size" : trainConfig.output_size,
    "hidden_size" : trainConfig.hidden_size,
    "all_words" : wordlist,
    "tags" : tags
}

FILE = "dataset/data.pth"
torch.save(data,FILE)

print(f'training complete . file saved to {FILE}')