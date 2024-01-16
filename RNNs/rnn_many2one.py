import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import glob
from tqdm import tqdm

import unicodedata
import string

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


"""
Data preparation
"""
class TextDataset(Dataset):
    """
    This class is in charge of managing text data as vectors
    Data is saved as vectors (not as text)
    """
    def __init__(self, data: dict, classes: list) -> None:
        self.data = data
        self.classes = classes
        self.num_classes = len(classes)
        self.vocab = self.concatenate_into_text(data)
        self.id2char = {u:v for u, v in enumerate(self.vocab)}
        self.char2id = {v:u for u, v in enumerate(self.vocab)}
        self.label2id = {v:u for u, v in enumerate(self.classes)}
        self.all_samples = []
        for class_name, samples in self.data.items():
            for sample in samples:
                self.all_samples.append((sample, class_name))
        self.data_size = len(self.all_samples)
        self.all_samples = [(self.string2vector(sample[0]), self.label2vector(sample[1])) for sample in self.all_samples]
        
    def string2vector(self, text: str) -> list:
        vector = []
        for c in text:
            vector.append(self.char2id[c])
        
        while len(vector) < 30:
            vector.append(self.char2id[" "])
        
        return vector
    
    def label2vector(self, text: str) -> list:
        vector = [0] * self.num_classes
        vector[self.label2id[text]] = 1
        return vector
    
    def concatenate_into_text(self, data: dict) -> str:
        """
        This function concatenates all label and sample into a single string
        Then return a list of unique characters of that string.
        """
        text = ""
        for key, value in data.items():
            text += key
            for vl in value:
                text += vl
        return sorted(list(set(text)))
    
    def __len__(self) -> int:
        """
        Number of samples
        """
        return self.data_size
    
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        X = torch.Tensor(self.all_samples[index][0]).float()
        Y = torch.Tensor(self.all_samples[index][1]).float()
        return X, Y


"""
Model definition
"""
class RNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, batch_size: int=1) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x, hidden_state) -> torch.Tensor:
        # Set initial hidden state
        
        x = self.i2h(x)
        hidden_state = self.h2h(hidden_state)
        hidden_state = torch.tanh(x + hidden_state)
        out = self.h2o(hidden_state)
        return nn.Softmax(dim=1)(out), hidden_state
    
    def init_zero_hidden(self, batch_size: int=-1) -> torch.Tensor:
        if batch_size == -1:
            batch_size = self.batch_size
        return torch.zeros(batch_size, self.hidden_size, requires_grad=False)
    

def train(model: RNN, data: DataLoader, epochs: int, optimizer: optim.Optimizer, loss_fn: nn.Module) -> None:
    train_losses = {}
    model.to(device)
    model.train()
    epoch_losses = list()
    print("==> Start training")
    
    for epoch in range(epochs):
        for X, Y in data:
            if X.shape[0] != model.batch_size:
                continue
            
            hidden = model.init_zero_hidden()
            X, Y, hidden = X.to(device), Y.to(device), hidden.to(device)
            model.zero_grad()
            
            for c in range(X.shape[1]):
                out, hidden = model(X[:, c].reshape(X.shape[0], 1), hidden)
            
            loss = loss_fn(out, Y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            epoch_losses.append(loss.detach().item() / X.shape[1])
            
        train_losses[epoch] = torch.Tensor(epoch_losses).mean()
        print(f'=> epoch: {epoch + 1}, loss: {train_losses[epoch]}')
        


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def listFiles(path):
    return glob.glob(path) 

def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

if __name__ == "__main__":
    
    data_path = "data/names/*.txt"
    
    all_letters = string.ascii_letters + " .,;'"
    n_letters = len(all_letters)

    category_lines = {}
    all_categories = []
    
    for filename in listFiles(data_path):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines
    
    
    batch_size = 32
    hidden_size = 64
    
    text_dataset = TextDataset(category_lines, all_categories)
    
    train_size = int(len(text_dataset) * 0.8)
    test_size = len(text_dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(text_dataset, [train_size, test_size])
    
    train_dataloader = DataLoader(train_set, batch_size)
    test_dataloader = DataLoader(test_set, batch_size)
    
    model = RNN(input_size=1, hidden_size=hidden_size, num_classes=text_dataset.num_classes, batch_size=batch_size)
    
    epochs = 100
    loss = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.01)
    
    train(model, train_dataloader, epochs, optimizer, loss)
    

    