import torch

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
