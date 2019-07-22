import librosa
import pandas as pd
from scipy.stats import skew
import scipy
from xgboost import XGBClassifier
import joblib
import os
import numpy as np

clf = XGBClassifier(max_depth=5,
                           min_child_weight=1,
                           learning_rate=0.1,
                           n_estimators=1000,
                           silent=True,
                           objective='binary:logistic',
                           gamma=0,
                           max_delta_step=0,
                           subsample=1,
                           colsample_bytree=1,
                           colsample_bylevel=1,
                           reg_alpha=0,
                           reg_lambda=0,
                           scale_pos_weight=1,
                           seed=1,
                           missing=None)
data = np.load(os.path.join('mfcc_approach', 'mfcc_features.npy'))
y = np.load(os.path.join('mfcc_approach', 'labels.npy'))

clf.fit(data, y)
joblib.dump(clf, 'xgb.dat')
