import os
import sys
import tqdm
from torch.autograd import Variable
import torch
import numpy as np
import pandas as pd
import librosa
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from model import SpoofDetector
from mfcc import mfcc_load


if __name__ == '__main__':
    model = SpoofDetector(input_shape=(50, 130))
    model.load_state_dict(torch.load('mfcc_model.dat'))
    model.cuda()
    model.eval()

    dataset_dir = "."
    eval_protocol_path = "protocol_test.txt"
    eval_protocol = pd.read_csv(eval_protocol_path, sep=" ", header=None)
    eval_protocol.columns = ['path', 'key']
    eval_protocol['score'] = 0.0

    print(eval_protocol.shape)
    print(eval_protocol.sample(10).head())

    for protocol_id, protocol_row in tqdm.tqdm(list(eval_protocol.iterrows())):
        x = mfcc_load(os.path.join(dataset_dir, protocol_row['path']))[None][None]
        x = Variable(torch.Tensor(x)).cuda()
        score = model(x).cpu().data.numpy()
        eval_protocol.at[protocol_id, 'score'] = 0 if score[0][0] < 0.5 else 1
    eval_protocol[['path', 'score']].to_csv('answers.csv', index=None)
    print(eval_protocol.sample(10).head())
    eval_protocol['key'] = eval_protocol['path'].apply(lambda x: 0 if 'spoof' in x else 1)
    print('Accuracy score: ' + str(accuracy_score(eval_protocol['key'], eval_protocol['score'])))
    print('ROC-AUC score: ' + str(roc_auc_score(eval_protocol['key'], eval_protocol['score'])))
    tn, fp, fn, tp = confusion_matrix(eval_protocol['key'], eval_protocol['score']).ravel()
    print('ERR rate: ' + str((fp + fn) / (tp + tn + fn + fp)))