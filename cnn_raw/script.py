import os
import sys
import tqdm
from torch.autograd import Variable
import torch
import numpy as np
import pandas as pd
import librosa
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import roc_curve
from model import SpoofNNDetector
SAMPLE_RATE = 22050


def eer_rate(y, y_pred):
    fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=1)
    fnr = 1 - tpr
    return fnr[np.nanargmin(np.absolute((fnr - fpr)))]

def get_features(path, sec=3.0, fr=SAMPLE_RATE):
    aud, _ = librosa.core.load(path, sr = SAMPLE_RATE)
    assert _ == SAMPLE_RATE
    if len(aud) < sec * fr:
        diff = int(sec * fr - len(aud))
        pad = np.random.randint(diff)
        aud = np.pad(aud, (pad, diff - pad), mode='constant')
    else:
        aud = aud[:int(sec * fr)]
    max_aud = np.max(np.abs(aud))
    min_aud = np.min(np.abs(aud))
    result = (aud - min_aud) / (max_aud - min_aud + 1e-6) - 0.5
    return result.astype('float64')

if __name__ == '__main__':
    model = SpoofNNDetector(input_size=66150).cuda()
    model.load_state_dict(torch.load('nn.dat'))
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
        x = get_features(os.path.join(dataset_dir, protocol_row['path']))[None][None]
        x = Variable(torch.Tensor(x)).cuda()
        score = model(x).cpu().data.numpy()
        eval_protocol.at[protocol_id, 'score'] = 0 if score[0][0] < 0.5 else 1
    eval_protocol[['path', 'score']].to_csv('answers.csv', index=None)
    print(eval_protocol.sample(10).head())
    eval_protocol['key'] = eval_protocol['path'].apply(lambda x: 0 if 'spoof' in x else 1)
    # print('Accuracy score: ' + str(accuracy_score(eval_protocol['key'], eval_protocol['score'])))
    # print('ROC-AUC score: ' + str(roc_auc_score(eval_protocol['key'], eval_protocol['score'])))
    # tn, fp, fn, tp = confusion_matrix(eval_protocol['key'], eval_protocol['score']).ravel()
    # print('ERR rate: ' + str((fp + fn) / (tp + tn + fn + fp)))
    # print('EER: ' + str(eer_rate(eval_protocol['key'], eval_protocol['score'])))
