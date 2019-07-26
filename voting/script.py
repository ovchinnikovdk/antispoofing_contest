import os
import sys
import tqdm
import joblib
# from joblib import Parallel, delayed
import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
import librosa
import sys
from models import SpoofDetector, FFTSpectogram, SpoofNNDetector
import torch
from torch.autograd import Variable
import multiprocessing as mp
from scipy.stats import skew


def preprocess(path, SAMPLE_RATE=22050, sec = 3.0, _n_fft=512):
    b, _ = librosa.core.load(path, sr=SAMPLE_RATE)
    assert _ == SAMPLE_RATE
    # ft1 = librosa.feature.mfcc(b, sr=SAMPLE_RATE, n_mfcc=20)
    # ft2 = librosa.feature.zero_crossing_rate(b)[0]
    # ft3 = librosa.feature.spectral_rolloff(b)[0]
    # ft4 = librosa.feature.spectral_centroid(b)[0]
    # ft5 = librosa.feature.spectral_contrast(b)[0]
    # ft6 = librosa.feature.spectral_bandwidth(b)[0]
    # ft1_trunc = np.hstack((np.mean(ft1, axis=1), np.std(ft1, axis=1), skew(ft1, axis = 1), np.max(ft1, axis = 1), np.min(ft1, axis = 1)))
    # ft2_trunc = np.hstack((np.mean(ft2), np.std(ft2), skew(ft2), np.max(ft2), np.min(ft2)))
    # ft3_trunc = np.hstack((np.mean(ft3), np.std(ft3), skew(ft3), np.max(ft3), np.min(ft3)))
    # ft4_trunc = np.hstack((np.mean(ft4), np.std(ft4), skew(ft4), np.max(ft4), np.min(ft4)))
    # ft5_trunc = np.hstack((np.mean(ft5), np.std(ft5), skew(ft5), np.max(ft5), np.min(ft5)))
    # ft6_trunc = np.hstack((np.mean(ft6), np.std(ft6), skew(ft6), np.max(ft6), np.min(ft6)))
    aud = b.copy()
    if len(aud) < 5.0 * SAMPLE_RATE:
        diff = int(5.0 * SAMPLE_RATE - len(aud))
        pad = np.random.randint(diff)
        aud = np.pad(aud, (pad, diff - pad), mode='constant')
    else:
        aud = aud[:int(5.0 * SAMPLE_RATE)]
    cqt = librosa.feature.chroma_cqt(y=aud, sr=SAMPLE_RATE)
    #FOR CNN
    if len(b) < sec * SAMPLE_RATE:
        diff = int(sec * SAMPLE_RATE - len(b))
        pad = np.random.randint(diff)
        b = np.pad(b, (pad, diff - pad), mode='reflect')
    else:
        b = b[:int(sec * SAMPLE_RATE)]
    b = b / np.max(np.abs(b))
    #STFT
    stft = librosa.stft(b, n_fft=_n_fft, hop_length=_n_fft, center=True)
    stft = librosa.amplitude_to_db(np.abs(stft), ref=np.max).astype('float64')
    #MFCC
    mfcc = librosa.feature.mfcc(b, sr=SAMPLE_RATE, n_mfcc=50)
    # mel = librosa.feature.melspectrogram(y=b, sr=SAMPLE_RATE, n_mels=128, fmax=8000)
    # cqt = librosa.power_to_db(mel, ref=np.max).astype('float64')
    # cens = librosa.feature.chroma_cens(y=b, sr=SAMPLE_RATE)

    return cqt.ravel(), mfcc, stft
    # return np.hstack((ft1_trunc, ft2_trunc, ft3_trunc, ft4_trunc, ft5_trunc, ft6_trunc))[None], mfcc, stft


if __name__ == '__main__':
    #cqt = joblib.load('cqt.dat')
    cqt = SpoofNNDetector(input_size=12*216, num_layers=8)
    cqt.load_state_dict(torch.load('nn.dat'))
    cqt.cuda()
    cqt.eval()
    fft = FFTSpectogram(input_shape=(257, 130))
    fft.load_state_dict(torch.load('cnn_fft_model.dat'))
    fft.cuda()
    fft.eval()
    cnn = SpoofDetector(input_shape=(50, 130))
    cnn.load_state_dict(torch.load('mfcc_model.dat'))
    cnn.cuda()
    cnn.eval()

    dataset_dir = "."
    eval_protocol_path = "protocol_test.txt"
    eval_protocol = pd.read_csv(eval_protocol_path, sep=" ", header=None)
    eval_protocol.columns = ['path', 'key']
    eval_protocol['score'] = 0.0
    eval_protocol['cqt'] = 0.0
    eval_protocol['fft'] = 0.0
    eval_protocol['cnn'] = 0.0

    print(eval_protocol.shape)
    print(eval_protocol.sample(20).head())
    print(eval_protocol.shape)

    eval_protocol['path_dir'] = eval_protocol['path'].apply(lambda x: os.path.join(dataset_dir, x))
    print('Preprocessing...')
    eval_protocol['preprocess'] = joblib.Parallel(n_jobs=mp.cpu_count())(joblib.delayed(preprocess)(row) for row in tqdm.tqdm(eval_protocol['path_dir']))
    print('Predicting...')
    for protocol_id, protocol_row in tqdm.tqdm(list(eval_protocol.iterrows())):
        #cqtoost
        x = protocol_row['preprocess'][0][None][None]
        x = Variable(torch.Tensor(x)).cuda()
        score_cqt = cqt(x).cpu().data.numpy()[0][0]
        score_cqt = 0 if score_cqt < 0.5 else 1
        eval_protocol.at[protocol_id, 'cqt'] = score_cqt
        #FFT
        x = protocol_row['preprocess'][2][None][None]
        x = Variable(torch.Tensor(x)).cuda()
        score_fft = fft(x).cpu().data.numpy()[0][0]
        score_fft = 0 if score_fft < 0.5 else 1
        eval_protocol.at[protocol_id, 'fft'] = score_fft
        #CNN
        x = protocol_row['preprocess'][1][None][None]
        x = Variable(torch.Tensor(x)).cuda()
        score_cnn = cnn(x).cpu().data.numpy()[0][0]
        score_cnn = 0 if score_cnn < 0.5 else 1
        eval_protocol.at[protocol_id, 'cnn'] = score_cnn
        #Voting
        eval_protocol.at[protocol_id, 'score'] = np.median([score_cqt, score_fft, score_cnn])
    print(eval_protocol[['path', 'cqt', 'fft', 'cnn', 'score']].sample(20).head())
    eval_protocol[['path', 'score']].to_csv('answers.csv', index=None)
