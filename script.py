import os
import sys
import tqdm
import joblib
import numpy as np
import pandas as pd
import librosa
from xgboost import XGBClassifier
import sys
sys.path.insert(0, './mfcc_approach')
from mfcc import get_features

if __name__ == '__main__':
    model = joblib.load('xgb.dat')

    dataset_dir = "."
    eval_protocol_path = "protocol_test.txt"
    eval_protocol = pd.read_csv(eval_protocol_path, sep=" ", header=None)
    eval_protocol.columns = ['path', 'key']
    eval_protocol['score'] = 0.0

    print(eval_protocol.shape)
    print(eval_protocol.sample(10).head())

    for protocol_id, protocol_row in tqdm.tqdm(list(eval_protocol.iterrows())):
        x = get_features(os.path.join(dataset_dir, protocol_row['path']))[None]
        score = model.predict(x)[0]
        eval_protocol.at[protocol_id, 'score'] = score
    eval_protocol[['path', 'score']].to_csv('answers.csv', index=None)
    print(eval_protocol.sample(10).head())
