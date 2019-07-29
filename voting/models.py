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
                                        torch.nn.Conv2d(10, 12, kernel_size=3, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(kernel_size=2, padding=0, stride=1),
                                        torch.nn.BatchNorm2d(12),
                                        torch.nn.Conv2d(12, 15, kernel_size=3, stride=1, padding=1),
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
                                        torch.nn.Conv2d(22, 27, kernel_size=3, padding=1, stride=1),
                                        torch.nn.LeakyReLU(),
                                        torch.nn.MaxPool2d(kernel_size=2, padding=0, stride=1),
                                        torch.nn.BatchNorm2d(27))
        self.linsize = (self.input_shape[0] - 6) * (self.input_shape[1] - 6)
        self.fc = torch.nn.Sequential(torch.nn.Linear(27 * self.linsize, 40),
                                     torch.nn.Dropout(0.6),
                                     torch.nn.Linear(40, 1),
                                     torch.nn.Sigmoid())


    def forward(self, input):
        out = self.conv(input)
        out = out.view(-1, 27 * self.linsize)
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
