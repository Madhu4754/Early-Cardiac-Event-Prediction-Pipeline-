import numpy as np

def compute_hrv(rr_intervals):
    # basic time-domain HRV metrics
    diffs = np.diff(rr_intervals)
    sdnn = np.std(rr_intervals)
    rmssd = np.sqrt(np.mean(diffs**2))
    return {'sdnn': sdnn, 'rmssd': rmssd}

def extract_trend_features(signal_windows):
    # difference of means to capture short-term trend
    means = [np.mean(w) for w in signal_windows]
    trend = np.diff(means).tolist()
    return {'trend_mean_diff': np.mean(trend) if len(trend)>0 else 0}
