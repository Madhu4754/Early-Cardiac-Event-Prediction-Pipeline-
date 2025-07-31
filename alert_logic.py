import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import argparse

def load_and_evaluate(model_path, scaler_path, data_path):
    from utils import load_model
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    import ast
    df['leads']=df['leads'].apply(ast.literal_eval)
    # featurize similarly to training (simplified)
    from model_training import featurize
    feats = pd.DataFrame([featurize(r) for _,r in df.iterrows()])
    y = feats['label']
    X = feats.drop(columns=['label']).fillna(0)
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[:,1]
    threshold = 0.5
    preds = (proba >= threshold).astype(int)
    print('Alert Threshold:', threshold)
    print('Classification Report:')
    print(classification_report(y, preds))
    print('Confusion Matrix:')
    print(confusion_matrix(y, preds))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--scaler', required=True)
    args=parser.parse_args()
    load_and_evaluate(args.model, args.scaler, args.data)
