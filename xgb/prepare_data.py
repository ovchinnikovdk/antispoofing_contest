import numpy as np
import librosa
import os
import tqdm
from multiprocessing import Pool
from scipy.stats import skew
import scipy
SAMPLE_RATE = 22050

def prepare_features(path, size=None, n_jobs=4):
    files, labels = filename_loader(path, balanced=False)
    if size is None:
        size = len(files)
    pool = Pool(n_jobs)
    data = pool.map(get_features, files[:size])
    np.save('mfcc_features', data)
    np.save('labels', np.array(labels[:size]))

def get_features(path):
    b, _ = librosa.core.load(path, sr = SAMPLE_RATE)
    assert _ == SAMPLE_RATE
    ft1 = librosa.feature.mfcc(b, sr = SAMPLE_RATE, n_mfcc=20)
    ft2 = librosa.feature.zero_crossing_rate(b)[0]
    ft3 = librosa.feature.spectral_rolloff(b)[0]
    ft4 = librosa.feature.spectral_centroid(b)[0]
    ft5 = librosa.feature.spectral_contrast(b)[0]
    ft6 = librosa.feature.spectral_bandwidth(b)[0]
    ft1_trunc = np.hstack((np.mean(ft1, axis=1), np.std(ft1, axis=1), skew(ft1, axis = 1), np.max(ft1, axis = 1), np.min(ft1, axis = 1)))
    ft2_trunc = np.hstack((np.mean(ft2), np.std(ft2), skew(ft2), np.max(ft2), np.min(ft2)))
    ft3_trunc = np.hstack((np.mean(ft3), np.std(ft3), skew(ft3), np.max(ft3), np.min(ft3)))
    ft4_trunc = np.hstack((np.mean(ft4), np.std(ft4), skew(ft4), np.max(ft4), np.min(ft4)))
    ft5_trunc = np.hstack((np.mean(ft5), np.std(ft5), skew(ft5), np.max(ft5), np.min(ft5)))
    ft6_trunc = np.hstack((np.mean(ft6), np.std(ft6), skew(ft6), np.max(ft6), np.min(ft6)))
    return np.hstack((ft1_trunc, ft2_trunc, ft3_trunc, ft4_trunc, ft5_trunc, ft6_trunc))

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
    prepare_features(os.path.join(os.pardir, os.path.join('data', 'train')), size=10000)
