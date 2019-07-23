import os
import sys
import tqdm
import joblib
import numpy as np
import pandas as pd
import librosa
from xgboost import XGBClassifier
import sys
from prepare_data import get_features
import multiprocessing as mp
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import RidgeClassifier

def preprocess(protocol_row):
    x = get_features(protocol_row)[None]
    return x

if __name__ == '__main__':
    model = joblib.load('xgb.dat')

    dataset_dir = "."
    eval_protocol_path = "protocol_test.txt"
    eval_protocol = pd.read_csv(eval_protocol_path, sep=" ", header=None)
    eval_protocol.columns = ['path', 'key']
    eval_protocol['score'] = 0.0

    print(eval_protocol.shape)
    print(eval_protocol.sample(10).head())
    print(eval_protocol.shape)

    p = mp.Pool(mp.cpu_count())
    eval_protocol['path_dir'] = eval_protocol['path'].apply(lambda x: os.path.join(dataset_dir, x))
    eval_protocol['preprocess'] = p.map(preprocess, eval_protocol['path_dir'])
    for protocol_id, protocol_row in tqdm.tqdm(list(eval_protocol.iterrows())):
        score = model.predict(protocol_row['preprocess'])[0]
        score_ridge = ridge.predict(protocol_row['preprocess'])
        eval_protocol.at[protocol_id, 'score'] = score
    eval_protocol[['path', 'score']].to_csv('answers.csv', index=None)
    print(eval_protocol.sample(10).head())
    eval_protocol['key'] = eval_protocol['path'].apply(lambda x: 0 if 'spoof' in x else 1)
    print('Accuracy score: ' + str(accuracy_score(eval_protocol['key'], eval_protocol['score'])))
    print('ROC-AUC score: ' + str(roc_auc_score(eval_protocol['key'], eval_protocol['score'])))
    tn, fp, fn, tp = confusion_matrix(eval_protocol['key'], eval_protocol['score']).ravel()
    print('ERR rate: ' + str((fp + fn) / (tp + tn + fn + fp)))
