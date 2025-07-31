import numpy as np
import pandas as pd
import argparse
import os

def generate_ecg_lead(length=500, noise=0.02, event=False):
    t = np.linspace(0,1,length)
    base = np.sin(2 * np.pi * 5 * t)
    qrs = np.exp(-((t - 0.5)**2)/(2*(0.01)**2)) * 5
    signal = base + qrs
    if event:
        # subtle pre-event increase in variability and baseline drift
        drift = 0.2 * np.sin(2 * np.pi * 2 * t)
        variability = 0.3 * np.sin(2 * np.pi * 50 * t) * np.random.uniform(0.5,1.0)
        signal += drift + variability
    signal += np.random.normal(0, noise, size=length)
    return signal

def build_dataset(n_samples=500, event_ratio=0.1, lookahead=3, output_path='sample_data/cardiac_dataset.csv', n_leads=2):
    n_event = int(n_samples * event_ratio)
    n_normal = n_samples - n_event
    data=[]
    for _ in range(n_normal):
        leads = [generate_ecg_lead(event=False) for _ in range(n_leads)]
        # no event in window and following lookahead windows
        label=0
        data.append({'leads': [l.tolist() for l in leads], 'label': label})
    for _ in range(n_event):
        # event occurs within lookahead window -> create prior windows with event flag for early warning
        leads = [generate_ecg_lead(event=True) for _ in range(n_leads)]
        label=1
        data.append({'leads': [l.tolist() for l in leads], 'label': label})
    df=pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f'Dataset saved to {output_path}. Events: {n_event}, Normal: {n_normal}')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output', default='sample_data/cardiac_dataset.csv')
    parser.add_argument('--n', type=int, default=500)
    parser.add_argument('--ratio', type=float, default=0.1)
    parser.add_argument('--leads', type=int, default=2)
    args=parser.parse_args()
    build_dataset(n_samples=args.n, event_ratio=args.ratio, output_path=args.output, n_leads=args.leads)
