import librosa
import pandas as pd
from scipy.stats import skew
import scipy
from xgboost import XGBClassifier
import joblib
import os
import numpy as np
# from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC

clf = XGBClassifier(max_depth=5,
                           min_child_weight=4,
                           learning_rate=0.1,
                           n_estimators=1000,
                           silent=True,
                           objective='binary:logistic',
                           gamma=0.1,
                           max_delta_step=0,
                           subsample=0.8,
                           colsample_bytree=0.9,
                           colsample_bylevel=1,
                           reg_alpha=0.005,
                           reg_lambda=0,
                           scale_pos_weight=1,
                           seed=13,
                           missing=None,
                           n_jobs=-1)



data = np.load('mfcc_features.npy')
y = np.load('labels.npy')

clf.fit(data, y)
joblib.dump(clf, 'xgb.dat')

# ridge = RidgeClassifier()
# ridge.fit(data, y)
# joblib.dump(ridge, 'ridge.dat')

svc = SVC(kernel='rbf')
svc.fit(data, y)
joblib.dump(svc, 'svc.dat')
