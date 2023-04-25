import torch.nn as nn


class LSTMModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers,
                            dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 50)
        self.fc2 = nn.Linear(50, 1)
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        y = self.fc1(h[self.num_layers - 1])
        y = self.relu(y)
        logits = self.fc2(y)
        # y = self.sigmoid(y)
        return logits


class LSTMMultiModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout, device):
        super().__init__()
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers,
                            dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 50)
        self.fc2 = {'ly16': nn.Linear(50, 1).to(device),
                    'ly555': nn.Linear(50, 1).to(device),
                    'REGN33': nn.Linear(50, 1).to(device),
                    'REGN87': nn.Linear(50, 1).to(device)}  # Predict only one value
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x, antibody):
        out, (h, c) = self.lstm(x)
        y = self.fc1(h[self.num_layers - 1])
        y = self.relu(y)
        logits = self.fc2[antibody](y)
        # y = self.sigmoid(y)
        return logits
