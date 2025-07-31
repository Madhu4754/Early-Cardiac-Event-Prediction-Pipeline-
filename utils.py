import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)

def sliding_window_eval(model, data_X, data_y, window_size=3):
    # simulate real-time sliding window by evaluating last `window_size` samples
    preds=[]
    for i in range(window_size, len(data_X)+1):
        window = data_X[i-window_size:i]
        proba = model.predict_proba(window)[:,1].mean()
        preds.append(1 if proba>=0.5 else 0)
    # align y
    true = data_y[window_size-1:]
    return {'preds': preds, 'true': true, 'report': classification_report(true, preds, output_dict=True)}
