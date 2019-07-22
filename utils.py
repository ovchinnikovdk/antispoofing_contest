import numpy as np
import librosa
import torch
from torch.autograd import Variable
from torch.optim import Adam
import sys
import os
from tqdm import tqdm
import time


def filename_loader(path, size=None, balanced=True):
    spoof_dir = os.path.join(path, 'spoof')
    human_dir = os.path.join(path, 'human')
    spoof_files, human_files = [], []
    sp_lst = os.listdir(spoof_dir)
    hu_lst = os.listdir(human_dir)
    if balanced:
        max_len = min(len(sp_lst), len(hu_lst))
        spoof_idx = np.array(range(len(sp_lst)))
        human_idx = np.array(range(len(hu_lst)))
        np.random.shuffle(spoof_idx)
        np.random.shuffle(human_idx)
        spoof_files = [os.path.join(spoof_dir, sp_lst[idx]) for idx in spoof_idx[:max_len]]
        human_files = [os.path.join(human_dir, hu_lst[idx]) for idx in human_idx[:max_len]]
    else:
        spoof_files = [os.path.join(spoof_dir, filename) for filename in sp_lst]
        human_files = [os.path.join(human_dir, filename) for filename in hu_lst]
    files = np.concatenate((spoof_files, human_files))
    labels = np.concatenate((np.array([1] * len(spoof_files)), np.zeros(len(human_files))))
    idx = np.array(range(len(files)))
    np.random.shuffle(idx)
    if size is None:
        size = len(files)
    return [files[idx[i]] for i in range(len(idx[:size]))], [labels[idx[i]] for i in range(len(idx[:size]))]


def sound_to_mel(sound):
    """Convert Sound to MEL-Spectogram"""
    return 1 + np.log(1.e-12 + librosa.feature.melspectrogram(sound[0], sr=sound[1], n_fft=1024, hop_length=256, fmin=20, fmax=8000, n_mels=80)).T / 10.


def dataloader(path, size=None, reshape=True):
    files, labels = filename_loader(path, size)
    assert len(files) == len(labels), 'len(labels) should be equal to len(files)'
    if reshape:
        return np.array([reshape_mel(sound_to_mel(librosa.load(file))) for file in files]), np.array(labels)
    else:
        return np.array([sound_to_mel(librosa.load(file)) for file in files]), np.array(labels)


def reshape_mel(mel, shape=(80, 80)):
    """Reshape MEL-spectogram Using Simple Method Pad/Trim"""
    if mel.shape[0] > shape[0]:
        diff = mel.shape[0] - shape[0]
        offset = np.random.randint(diff)
        mel = mel[offset:shape[0] + offset, :]
    elif mel.shape[0] < shape[0]:
        diff = shape[0] - mel.shape[0]
        offset = np.random.randint(diff)
        mel = np.pad(mel, ((offset, shape[0] - mel.shape[0] - offset), (0, 0)), "reflect")
    if mel.shape[1] > shape[1]:
        diff = mel.shape[1] - shape[1]
        offset = np.random.randint(diff)
        mel = mel[:, offset:shape[1] + offset]
    elif mel.shape[1] < shape[1]:
        diff = shape[1] - mel.shape[1]
        offset = np.random.randint(diff)
        mel = np.pad(mel, ((0, 0), (offset, shape[1] - mel.shape[1] - offset)), "reflect")
    return mel


def batch_iterator(path, batch_size=256, shuffle=False):
    data = np.load(os.path.join(path, 'train.npy'))
    labels = np.load(os.path.join(path, 'labels.npy'))
    idx = np.array(range(len(data)))
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        x = np.stack([data[idx[j]][None] for j in range(i, min(i + batch_size, len(data)))], 0)
        y = np.stack([labels[idx[j]] for j in range(i, min(i + batch_size, len(data)))], 0)
        yield x, y



class SpoofDetector(torch.nn.Module):
    """Spoof Detector Convolutional Neural network"""
    def __init__(self, input_shape=(80, 80)):
        super(SpoofDetector, self).__init__()
        # input_shape = (80, 80)
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


def train(net, path, batch_size=128, n_epochs=30, lr=1e-4):
    optimizer = Adam(net.parameters(), lr=lr)
    loss = torch.nn.BCELoss()
    net.train()
    for i in tqdm(range(n_epochs), desc='Training epochs'):
        sum_loss = 0
        for x, y in batch_iterator(path, batch_size=batch_size):
            x = Variable(torch.Tensor(x)).cuda()
            y = Variable(torch.Tensor(y)).cuda()
            optimizer.zero_grad()
            output = net(x)
            loss_out = loss(output, y)
            loss_out.backward()
            optimizer.step()
            sum_loss += loss_out.data[0]
        if i % 5 == 0 and i > 0:
            print("EPOCH #" + str(i))
            print("Loss: " + str(sum_loss))
            if i % 10 == 0:
                torch.save(net, 'trained/cnn_epoch' + str(i) + '.pth')
    torch.save(net, 'trained/cnn.pth')
