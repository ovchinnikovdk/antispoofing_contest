import numpy as np
import librosa
import torch
from torch.autograd import Variable
from torch.optim import Adam
import sys
import os
from tqdm import tqdm
import time
from model import SpoofDetector
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve

def eer_rate(y, y_pred):
    fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=1)
    fnr = 1 - tpr
    return fnr[np.nanargmin(np.absolute((fnr - fpr)))]


def batch_iterator_mfcc(path, batch_size=256, shuffle=False):
    batches = np.load(path)
    labels = np.load(path.replace('data_', 'labels_'))
    for i in range(0, len(batches), batch_size):
        x = np.stack([batches[j][None] for j in range(i, min(i + batch_size, len(batches)))], 0)
        y = np.stack([labels[j][None] for j in range(i, min(i + batch_size, len(labels)))], 0)
        yield x, y


def train(net, path, batch_size=128, n_epochs=30, lr=1e-4):
    optimizer = Adam(net.parameters(), lr=lr)
    loss = torch.nn.BCELoss()
    net.train()
    train_paths = list(filter(lambda x : x.find('data_') != -1, os.listdir(path)))
    val_data = [x[None] for x in np.load(os.path.join(path, train_paths[0]))]
    val_data = Variable(torch.Tensor(val_data)).cuda()
    print('Val_shape: ',val_data.shape)
    val_y = np.load(os.path.join(path, train_paths[0].replace('data_', 'labels_')))
    train_paths = [os.path.join(path, tr) for tr in train_paths[1:3]]
    for i in tqdm(range(n_epochs), desc='Training epochs'):
        net.train()
        sum_loss = 0
        for tr_path in train_paths:
            for x, y in batch_iterator_mfcc(tr_path, batch_size=batch_size):
                x = Variable(torch.Tensor(x)).cuda()
                y = Variable(torch.Tensor(y)).cuda()
                optimizer.zero_grad()
                output = net(x)
                loss_out = loss(output, y)
                loss_out.backward()
                optimizer.step()
                sum_loss += loss_out.data[0]
        if i % 3 == 0 and i > 0:
            print("EPOCH #" + str(i))
            print("Loss: " + str(sum_loss))
            torch.cuda.empty_cache()
            net.eval()
            pred_y = []
            for val_x in val_data:
                pred = net(val_x[None]).cpu().data.numpy()[0][0]
                pred = 1 if pred >= 0.5 else 0
                pred_y.append(pred)
            print('Accuracy score: ' + str(accuracy_score(val_y, pred_y)))
            print('EER: ' + str(eer_rate(val_y, pred_y)))
            torch.save(net.state_dict(), 'trained/mfcc_model_e' + str(i) + '.dat')
    torch.save(net.state_dict(), 'mfcc_model.dat')

if __name__ == '__main__':
    model = SpoofDetector(input_shape=(50, 130)).cuda()
    path = os.path.join(os.pardir, 'data')
    path = os.path.join(path, 'mfcc_data')
    train(model, batch_size=70, n_epochs=16, path=path)
