import numpy as np
import librosa
import os
import tqdm
from multiprocessing import Pool
from joblib import Parallel, delayed
from scipy.stats import skew
import scipy
SAMPLE_RATE = 22050

def prepare_features(path, size=None, n_jobs=4):
    files, labels = filename_loader(path, balanced=True)
    if size is None:
        size = len(files)

    data = Parallel(n_jobs=n_jobs)(delayed(get_features)(filename) for filename in tqdm.tqdm(files[:size]))
    save_dir = os.path.join(os.pardir, 'data')
    save_dir = os.path.join(save_dir, 'features_cqt')
    np.save(os.path.join(save_dir, 'features'), np.array(data))
    np.save(os.path.join(save_dir, 'labels'), np.array(labels[:size]))

def get_features(path, sec=5.0, fr=SAMPLE_RATE):
    aud, _ = librosa.core.load(path, sr = SAMPLE_RATE)
    assert _ == SAMPLE_RATE
    if len(aud) < sec * fr:
        diff = int(sec * fr - len(aud))
        pad = np.random.randint(diff)
        aud = np.pad(aud, (pad, diff - pad), mode='constant')
    else:
        aud = aud[:int(sec * fr)]
    cqt = librosa.feature.chroma_cqt(y=aud, sr=SAMPLE_RATE)
    return cqt

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
    labels = np.concatenate((np.zeros(len(spoof_files)), np.array([1]*len(human_files))))
    idx = np.array(range(len(files)))
    np.random.shuffle(idx)
    if size is None:
        size = len(files)
    return [files[idx[i]] for i in range(len(idx[:size]))], [labels[idx[i]] for i in range(len(idx[:size]))]

if __name__ == '__main__':
    prepare_features(os.path.join(os.pardir, os.path.join('data', 'train')))
