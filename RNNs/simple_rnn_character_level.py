import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

"""
Data preparation
"""
class TextDataset(Dataset):
    """
    This class is in charge of managing text data as vectors
    Data is saved as vectors (not as text)
    
    Attributes
    ------------
    seq_length : int = Sequence length
    chars : list = list of characters
    id2char : dict = dictionary from id to character
    char2id : dict = dictionary from character to id
    vocab_size : int = vocab size
    data_size : int = total length of text data
    X : vector = vector form of text data
    """
    
    def __init__(self, text_data: str, seq_length: int=25) -> None:
        """
        Inputs
        ------------
        text_data: Full text data as string
        seq_length: how many characters per index of the dataset
        """
        while len(text_data) % seq_length != 0:
            text_data += " "
            
        self.vocab = sorted(list(set(text_data)))
        self.data_size, self.vocab_size = len(text_data), len(self.vocab)
        self.id2char = {u:v for u, v in enumerate(self.vocab)}
        self.char2id = {v:u for u, v in enumerate(self.vocab)}
        self.seq_length = seq_length
        self.X = self.string2vector(text_data)
        
        
        
    @property
    def X_string(self) -> str:
        """
        Return X in string form
        """
        return self.vector2string(self.X)

    def __len__(self) -> int:
        """
        Number of sequences (except the last one)
        """
        return int(len(self.X) / self.seq_length - 1)
    
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        """
        X and Y have the same shape, but X is shifted left 1 position
        """
        start_idx = index * self.seq_length
        end_idx = start_idx + self.seq_length
        
        X = torch.Tensor(self.X[start_idx: end_idx]).float()
        Y = torch.Tensor(self.X[start_idx+1: end_idx+1]).float()
        return X, Y
    
    def string2vector(self, text: str) -> list[int]:
        """
        Convert string to 1d vector
        """
        vector = []
        for s in text:
            vector.append(self.char2id[s])
        return vector
    
    def vector2string(self, vector: list[int]) -> str:
        """
        Convert 1d vector to string
        """
        text = ""
        for id in vector:
            text += self.id2char[id]
        return text
    
    
"""
Model definition
"""
class RNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, batch_size: int=1) -> None:
        """
        input_size: dimension of input vector
        hidden_size: dimension of hidden vector
        output_size: dimension of output vector
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        
        
    def forward(self, x, hidden_state) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return softmax(linear_out), tanh(i2h + i2o)
        
        Inputs
        -------
        x = Input vector x with shape (input_size, 1)
        hidden_state: Hidden state matrix
        
        Outputs
        -------
        out: Prediction vector
        hidden_state: New hidden state matrix
        """
        
        x = self.i2h(x)
        hidden_state = self.h2h(hidden_state)
        hidden_state = torch.tanh(x + hidden_state)
        
        return self.h2o(hidden_state), hidden_state
    
    def init_zero_hidden(self, batch_size: int=-1) -> torch.Tensor:
        """
        Returns a hidden state with specified batch size.
        """
        if batch_size == -1:
            batch_size = self.batch_size
        return torch.zeros(batch_size, self.hidden_size, requires_grad=False)
        

def generate_text(model: RNN, dataset: TextDataset, prediction_length: int=128) -> str:
    """
    Generate text up to prediction_length characters
    """
    
    model.eval()
    predicted = dataset.vector2string([random.randint(0, len(dataset.vocab)-1)])
    hidden = model.init_zero_hidden(batch_size=1) # 1 because we want to generate for 1 sample only
    for i in range(prediction_length - 1):
        last_char = torch.Tensor([dataset.char2id[predicted[-1]]])
        X, hidden = last_char.to(device), hidden.to(device)
        out, hidden = model(X, hidden)
        result = torch.multinomial(nn.functional.softmax(out, 1), 1).item()
        predicted += dataset.id2char[result]
    
    return predicted
        
        
def train(model: RNN, data: DataLoader, epochs: int, optimizer: optim.Optimizer, loss_fn: nn.Module) -> None:
    """
    Train the model for the specified number of epochs
    
    Inputs:
    --------
    model: RNN model to train
    data: Iterable DataLoader
    epochs: number of epochs to train the model
    optimizer: Optimizer to use 
    loss_fn: Function to calculate loss
    
    """
    
    train_losses = {}
    model.to(device)
    model.train()
    print("==> Start training")
    for epoch in range(epochs):
        epoch_losses = list()
        for X, Y in data:
            if X.shape[0] != model.batch_size:
                continue
                # raise Exception("Input shape does not fit the batch size")
            
            hidden = model.init_zero_hidden()
            
            X, Y, hidden = X.to(device), Y.to(device), hidden.to(device)
            
            model.zero_grad()
            
            loss = 0
            for c in range(X.shape[1]):
                out, hidden = model(X[:, c].reshape(X.shape[0], 1), hidden)
                l = loss_fn(out, Y[:, c].long())
                loss += l
                
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            
            epoch_losses.append(loss.detach().item() / X.shape[1])
            
        train_losses[epoch] = torch.Tensor(epoch_losses).mean()
        print(f'=> epoch: {epoch + 1}, loss: {train_losses[epoch]}')
        print(generate_text(model, data.dataset))
        
        
if __name__ == "__main__":
    data = open('data/text.txt', 'r').read() # use any text file you want to learn
    data = data.lower()

    # Data size variables
    seq_length = 128
    batch_size = 16
    hidden_size = 256

    text_dataset = TextDataset(data, seq_length=seq_length)
    text_dataloader = DataLoader(text_dataset, batch_size)
    
    # Model
    rnnModel = RNN(1, hidden_size, len(text_dataset.vocab), batch_size) # 1 because we enter a single number/letter per step.

    # Train variables
    epochs = 1000
    loss = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(rnnModel.parameters(), lr = 0.005)

    train(rnnModel, text_dataloader, epochs, optimizer, loss)