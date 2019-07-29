import torch
from torch.autograd import Variable

class SpoofNNDetector(torch.nn.Module):
    """Spoof Detector Neural network"""
    def __init__(self, input_size=20):
        super(SpoofNNDetector, self).__init__()
        self.input_size = input_size
        self.hidden = None
        self.conv = torch.nn.Sequential(
                                        torch.nn.Conv1d(1, 12, kernel_size=13),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool1d(kernel_size=7),
                                        torch.nn.BatchNorm1d(12),
                                        torch.nn.Conv1d(12, 15, kernel_size=13),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool1d(kernel_size=7),
                                        torch.nn.BatchNorm1d(15),
                                        torch.nn.Conv1d(15, 20, kernel_size=13),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool1d(kernel_size=7),
                                        torch.nn.BatchNorm1d(20),
        )
        self.lstm = torch.nn.LSTM(input_size=20 * 190,
                                    hidden_size=100,
                                    num_layers=5,
                                    batch_first=True,
                                    dropout=0.5,
                                    bidirectional=True)
        self.fc = torch.nn.Sequential(torch.nn.Linear(200, 50),
                                     torch.nn.Dropout(0.6),
                                     torch.nn.Linear(50, 1),
                                     torch.nn.Sigmoid()
                                     )



    def forward(self, input):
        out = self.conv(input)
        # print(out.shape)
        out = out.view(-1, 1, 20 * 190)
        out, hidden = self.lstm(out)
        # self.hidden = hidden
        out = out.view(-1, 200)
        out = self.fc(out)
        return out
