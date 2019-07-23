import torch

class SpoofNNDetector(torch.nn.Module):
    """Spoof Detector Neural network"""
    def __init__(self, input_size=20, hidden_size=200):
        super(SpoofNNDetector, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc = torch.nn.Sequential(torch.nn.Linear(input_size, hidden_size),
                                    torch.nn.Tanh(),
                                     torch.nn.Linear(hidden_size, 50),
                                     torch.nn.Tanh(),
                                     torch.nn.Linear(50, 1)
                                     )


    def forward(self, input):
        out = self.fc(input)
        return out
