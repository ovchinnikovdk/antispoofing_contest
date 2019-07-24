import numpy as np
import librosa
import os
import tqdm
from multiprocessing import Pool
import multiprocessing as mp


def mfcc_load(filename, fr = 22050, sec = 3.0):
    aud, _ = librosa.load(filename, sr=fr)
    if len(aud) < sec * fr:
        diff = int(sec * fr - len(aud))
        pad = np.random.randint(diff)
        aud = np.pad(aud, (pad, diff - pad), mode='reflect')
    else:
        aud = aud[:int(sec * fr)]
    # aud = aud / np.max(np.abs(aud))
    return np.abs(ibrosa.feature.mfcc(aud, sr=fr, n_mfcc=50))

def preprocess_batch(files, labels, num, n_jobs=mp.cpu_count()):
    assert len(files) == len(labels), 'files and labels should be same count.'

    pool = Pool(n_jobs)
    data = pool.map(mfcc_load, files)
    save_dir = os.path.join(os.pardir, 'data')
    save_dir = os.path.join(save_dir, 'mfcc_data')
    np.save(os.path.join(save_dir, 'data_batch_' + str(num)), np.array(data))
    np.save(os.path.join(save_dir, 'labels_batch_' + str(num)), np.array(labels))

def preprare_data(path, split_size=16, balanced=True):
    files, labels = filename_loader(path, balanced=balanced)
    batch_size = int(len(files) / split_size)
    for i in tqdm.tqdm(range(split_size)):
        preprocess_batch(files[batch_size*i:(i+1)*batch_size], labels[batch_size*i:(i+1)*batch_size], i)



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
    path = os.path.join(os.pardir, 'data')
    path = os.path.join(path, 'train')
    preprare_data(path)
