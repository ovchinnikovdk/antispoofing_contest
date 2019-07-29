import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve


def eer_rate(y, y_pred):
    fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=1)
    fnr = 1 - tpr
    return fnr[np.nanargmin(np.absolute((fnr - fpr)))]

eval_protocol = pd.read_csv('answers.csv')
eval_protocol['key'] = eval_protocol['path'].apply(lambda x: 0 if 'spoof' in x else 1)
print('---------------------')
print('Final voting => Accuracy score: ' + str(accuracy_score(eval_protocol['key'], eval_protocol['score'])))
print('Final voting => ROC-AUC score: ' + str(roc_auc_score(eval_protocol['key'], eval_protocol['score'])))
tn, fp, fn, tp = confusion_matrix(eval_protocol['key'], eval_protocol['score']).ravel()
print('Final voting => ERR rate: ' + str((fp + fn) / (tp + tn + fn + fp)))
print('Final voting => EER: ' + str(eer_rate(eval_protocol['key'], eval_protocol['score'])))
print('---------------------')
