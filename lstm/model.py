import torch
from torch.autograd import Variable

class SpoofNNDetector(torch.nn.Module):
    """Spoof Detector Neural network"""
    def __init__(self, input_size=20, n_chroma=12, hidden_size=200, dropout=0.5, num_layers=1):
        super(SpoofNNDetector, self).__init__()
        self.n_chroma = n_chroma
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_size=input_size,
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    batch_first=True,
                                    dropout=dropout,
                                    bidirectional=True)
        self.fc = torch.nn.Sequential(torch.nn.Linear(hidden_size * 2, 70),
                                     torch.nn.Dropout(0.5),
                                     torch.nn.Linear(70, 1)
                                     )



    def forward(self, input):
        out, _ = self.lstm(input)
        out = self.fc(out)
        return out
