import torch

class SpoofDetector(torch.nn.Module):
    """Spoof Detector Convolutional Neural network"""
    def __init__(self, input_shape=(80, 80)):
        super(SpoofDetector, self).__init__()
        self.input_shape = input_shape
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(kernel_size=2, padding=0, stride=1),
                                        torch.nn.BatchNorm2d(10),
                                        torch.nn.Conv2d(10, 15, kernel_size=3, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(kernel_size=2, padding=0, stride=1),
                                        torch.nn.BatchNorm2d(15),
                                        torch.nn.Conv2d(15, 20, kernel_size=3, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(kernel_size=2, padding=0, stride=1),
                                        torch.nn.Conv2d(20, 30, kernel_size=3, padding=1, stride=1),
                                        torch.nn.LeakyReLU(),
                                        torch.nn.MaxPool2d(kernel_size=2, padding=0, stride=1),
                                        torch.nn.BatchNorm2d(30))
        self.linsize = (self.input_shape[0] - 4) * (self.input_shape[1] - 4)
        self.fc = torch.nn.Sequential(torch.nn.Linear(30 * self.linsize, 20),
                                     torch.nn.Dropout(0.6),
                                     torch.nn.Linear(20, 1),
                                     torch.nn.Sigmoid())


    def forward(self, input):
        out = self.conv(input)
        out = out.view(-1, 30 * self.linsize)
        out = self.fc(out)
        return out


class FFTSpectogram(torch.nn.Module):
    """FFTSpectogram Convolutional Neural network"""
    def __init__(self, input_shape=(80, 80)):
        super(FFTSpectogram, self).__init__()
        self.input_shape = input_shape
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(kernel_size=2, padding=0, stride=1),
                                        torch.nn.BatchNorm2d(6),
                                        # torch.nn.Conv2d(10, 15, kernel_size=3, stride=1, padding=1),
                                        # torch.nn.ReLU(),
                                        # torch.nn.MaxPool2d(kernel_size=2, padding=0, stride=1),
                                        # torch.nn.BatchNorm2d(15),
                                        torch.nn.Conv2d(6, 8, kernel_size=3, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(kernel_size=2, padding=0, stride=1),
                                        torch.nn.Conv2d(8, 10, kernel_size=3, padding=1, stride=1),
                                        torch.nn.LeakyReLU(),
                                        torch.nn.MaxPool2d(kernel_size=2, padding=0, stride=1),
                                        torch.nn.BatchNorm2d(10))
        self.linsize = (self.input_shape[0] - 3) * (self.input_shape[1] - 3)
        self.fc = torch.nn.Sequential(torch.nn.Linear(10 * self.linsize, 15),
                                     torch.nn.Dropout(0.6),
                                     torch.nn.Linear(15, 1),
                                     torch.nn.Sigmoid())


    def forward(self, input):
        out = self.conv(input)
        out = out.view(-1, 10 * self.linsize)
        out = self.fc(out)
        return out


class CQTChromogram(torch.nn.Module):
    """CQTChromogram Convolutional Neural network"""
    def __init__(self, input_shape=(80, 80)):
        super(CQTChromogram, self).__init__()
        self.input_shape = input_shape
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(kernel_size=2, padding=0, stride=1),
                                        torch.nn.BatchNorm2d(8),
                                        torch.nn.Conv2d(8, 10, kernel_size=3, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(kernel_size=2, padding=0, stride=1),
                                        torch.nn.BatchNorm2d(10),
                                        torch.nn.Conv2d(10, 15, kernel_size=3, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(kernel_size=2, padding=0, stride=1),
                                        torch.nn.BatchNorm2d(15),
                                        torch.nn.Conv2d(15, 20, kernel_size=3, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(kernel_size=2, padding=0, stride=1),
                                        torch.nn.BatchNorm2d(20),
                                        torch.nn.Conv2d(20, 22, kernel_size=3, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(kernel_size=2, padding=0, stride=1),
                                        torch.nn.Conv2d(22, 24, kernel_size=3, padding=1, stride=1),
                                        torch.nn.LeakyReLU(),
                                        torch.nn.MaxPool2d(kernel_size=2, padding=0, stride=1),
                                        torch.nn.BatchNorm2d(24))
        self.linsize = (self.input_shape[0] - 6) * (self.input_shape[1] - 6)
        self.fc = torch.nn.Sequential(torch.nn.Linear(24 * self.linsize, 30),
                                     torch.nn.Dropout(0.6),
                                     torch.nn.Linear(30, 1),
                                     torch.nn.Sigmoid())


    def forward(self, input):
        out = self.conv(input)
        out = out.view(-1, 24 * self.linsize)
        out = self.fc(out)
        return out

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
        self.fc = torch.nn.Sequential(torch.nn.Linear(hidden_size * 2, 50),
                                     torch.nn.Dropout(0.6),
                                     torch.nn.Linear(50, 1)
                                     )



    def forward(self, input):
        out, _ = self.lstm(input)
        out = self.fc(out)
        return out
