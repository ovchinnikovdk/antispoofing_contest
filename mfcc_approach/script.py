import os
import sys
import tqdm
from torch.autograd import Variable
import torch
import numpy as np
import pandas as pd
import librosa

from model import SpoofDetector
from mfcc import mfcc_load
import utils


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
