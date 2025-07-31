# Early Cardiac Event Prediction Pipeline

## Overview
Simulated research project building an end-to-end pipeline to predict early cardiac events using multi-lead ECG segments. Derives Heart Rate Variability (HRV) and trend features, applies temporal windowing, and uses ensemble models for robust early warning detection. Includes sliding-window evaluation to mimic real-time monitoring and threshold-based clinical alert logic.

## Structure
- `data_generator.py`: Synthesizes multi-lead ECG signals including pre-event perturbations and normal patterns; creates synthetic onset labels with lookahead windows.
- `hrv_features.py`: Computes HRV metrics from inter-beat intervals and trend features.
- `preprocessing.py`: Signal filtering and normalization per lead.
- `model_training.py`: Builds ensemble of short-term and trend models; combines via weighted stacking and performs sliding window evaluation.
- `alert_logic.py`: Mock clinical alert thresholding and sensitivity/specificity tradeoff simulation.
- `utils.py`: Shared functions for evaluation, saving/loading, windowing, and reproducibility tracking.
- `notebooks/`: Example Jupyter notebook demonstrating pipeline and tracking experiments.
- `requirements.txt`: Dependencies list.
- `sample_data/`: Synthetic dataset with labels.
## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Generate data: `python data_generator.py --output sample_data/cardiac_dataset.csv`
3. Train models: `python model_training.py --data sample_data/cardiac_dataset.csv`
4. Simulate alerts: `python alert_logic.py --data sample_data/cardiac_dataset.csv`
## Goals
- Early detection of events via ensemble modeling and sliding-window validation.  
- Balance sensitivity/specificity with clinical-style alert thresholds.  
- Reproducible experiments via versioned logging.  

## License
MIT
