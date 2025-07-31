import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from hrv_features import compute_hrv, extract_trend_features
from preprocessing import bandpass_filter, normalize
from utils import evaluate_model, save_model
import argparse, os, joblib
from sklearn.metrics import roc_auc_score

def featurize(row):
    leads = row['leads']
    all_feats = {}
    # assume leads is list of lists
    for idx, lead in enumerate(leads):
        signal = np.array(lead)
        filtered = normalize(bandpass_filter(signal))
        # simple R-peak approximation: find peaks above threshold
        peaks = np.where(filtered > 0.8)[0]
        rr_intervals = np.diff(peaks) if len(peaks)>1 else np.array([0.0])
        hrv = compute_hrv(rr_intervals)
        # trend over sliding subwindows
        windowed = [filtered[i:i+50] for i in range(0, len(filtered)-50, 25)]
        trend = extract_trend_features(windowed)
        prefix = f'lead{idx}_'
        for k,v in {**hrv, **trend}.items():
            all_feats[prefix + k] = v
    all_feats['label'] = row['label']
    return all_feats

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output_dir', default='models')
    args=parser.parse_args()
    df=pd.read_csv(args.data)
    # convert stringified lists to actual lists
    import ast
    df['leads']=df['leads'].apply(ast.literal_eval)
    Xy = pd.DataFrame([featurize(r) for _,r in df.iterrows()])
    y = Xy['label']
    X = Xy.drop(columns=['label']).fillna(0)
    scaler=StandardScaler()
    X_scaled=scaler.fit_transform(X)
    os.makedirs(args.output_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(args.output_dir,'scaler.pkl'))
    X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,stratify=y,test_size=0.2,random_state=42)
    rf=RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=42)
    gb=GradientBoostingClassifier(n_estimators=100)
    rf.fit(X_train,y_train)
    gb.fit(X_train,y_train)
    # evaluate
    from sklearn.metrics import classification_report
    print('Random Forest report:')
    print(classification_report(y_test, rf.predict(X_test)))
    print('Gradient Boosting report:')
    print(classification_report(y_test, gb.predict(X_test)))
    save_model(rf, os.path.join(args.output_dir,'rf.pkl'))
    save_model(gb, os.path.join(args.output_dir,'gb.pkl'))
    print('Models saved.')
if __name__=='__main__':
    main()
