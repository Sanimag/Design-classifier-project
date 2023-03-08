import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
import sklearn 
from sklearn import preprocessing
from torch.utils.data import DataLoader

vocab = np.load("/content/drive/MyDrive/design_classfier/vocab.npy", allow_pickle=True)
weights_matrix = np.load("/content/drive/MyDrive/design_classfier/weights_matrix.npy", allow_pickle=True)

le = preprocessing.LabelEncoder()
le.fit(vocab)

class Dataset(torch.utils.data.Dataset):

    def __init__(self, dataset):

        #text = text_prep(text)
        self.data = dataset
        self.transform = transforms.Compose([ transforms.ToTensor()])
        
    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
    
        if not torch.is_tensor(self.data[idx]["image"]):
            self.data[idx]["image"] = self.transform(self.data[idx]["image"])
            self.data[idx]["image"] /= 255

        return (le.transform(self.data[idx]["text"].split()), self.data[idx]["image"], self.data[idx]["score"])

class LSTM(nn.Module):

    def __init__(self, weights_matrix, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.embedding, self.num_embeddings, self.embedding_dim = create_embed_layer(weights_matrix, True)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=False, proj_size=0)
        self.relu = nn.ReLU()
        self.fc_1 = nn.Linear(hidden_size, 64)
        self.fc = nn.Linear(64,hidden_size)

    def forward(self, x, h, c):
        
        #print("EMBEDDING INFO", self.num_embeddings, self.embedding_dim)
        #print(len(x), type(x))
        x = torch.Tensor(x)
        #print(x, x.shape)
        x = self.embedding(x)

        # Propagate input through LSTM
        output, (h, c) = self.lstm(x, (h, c)) #lstm with input, hidden, and internal state
        output = output.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(output)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out, h, c
        
    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size))            
    
class Validator(nn.Module):
                
    def __init__(self, weight_matrix, hidden_size, num_classes):
    
        super(Validator,self).__init__()
    
        self.cnn = AlexNet(num_classes=num_classes, out_size = hidden_size)
        self.lstm = LSTM(weights_matrix=weight_matrix, hidden_size=hidden_size, num_layers=1)
    
        self.fc1 = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)
        self.activation = nn.ReLU()

    def forward(self, text, image,h, c):

        image_output = self.cnn(image)
        text_output, h, c = self.lstm(text, h, c)
        
        #text_output = text_output.reshape((1, -1))
        #print(" text shpe before summmation : ", text_output.shape)
        text_output = torch.sum(text_output, dim=0)
        text_output = text_output.reshape((1, -1))

        #print("IMAGE SHAPE ", image_output.shape, " TEXT SHAPE ", text_output.shape)
        
        hidden = torch.cat(tensors=(image_output, text_output), dim=1)

        hidden = self.activation(hidden)
        hidden = self.fc1(hidden)
        hidden = self.activation(hidden)
        hidden = self.out(hidden)

        return hidden, h, c

class AlexNet(nn.Module):
    def __init__(self, num_classes=16, out_size=50):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 2048),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(2048, out_size))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

def create_embed_layer(weights_matrix, non_trainable=True):

    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': torch.Tensor(weights_matrix)})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

SCALER = sklearn.preprocessing.StandardScaler()
MEAN_SCALER = 0.50817122
STD_SCALER = 0.43327774 


net = Validator(weight_matrix=weights_matrix, hidden_size=50, num_classes=1)
