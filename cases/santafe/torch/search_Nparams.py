import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


# Versión Vanilla: Se proporciona como referencia
class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        #https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        super(VanillaRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        #print(out)#print(_)#print(out.shape)
        out = self.fc(out[:, -N:, :])
        #print(out.shape)
        return out

# Versión LSTM: A implementar
# Escribe el código de la versión LSTM del modelo
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        #https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -N:, :])
        return out

# Versión GRU: A implementar 
# Escribe el código de la versión GRU del modelo
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        #https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -N:, :])
        return out
    


for i in range(1,7):
    for j in range(1,2):
        vanilla_rnn = VanillaRNN(input_size=1, hidden_size=i, output_size=1, num_layers=j)
        Nparams = sum(p.numel() for p in vanilla_rnn.parameters())
        print('input_size=1, hidden_size=%i, output_size=1, num_layers=%i :  %i' %(i,j,Nparams))

print()
for i in range(1,6):
    for j in range(1,2):
        vanilla_rnn = LSTMModel(input_size=1, hidden_size=i, output_size=1, num_layers=j)
        Nparams = sum(p.numel() for p in vanilla_rnn.parameters())
        print('input_size=1, hidden_size=%i, output_size=1, num_layers=%i :  %i' %(i,j,Nparams))


print()
for i in range(1,6):
    for j in range(1,2):
        vanilla_rnn = GRUModel(input_size=1, hidden_size=i, output_size=1, num_layers=j)
        Nparams = sum(p.numel() for p in vanilla_rnn.parameters())
        print('input_size=1, hidden_size=%i, output_size=1, num_layers=%i :  %i' %(i,j,Nparams))