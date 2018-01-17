import pandas as pd
import numpy as np

LSTM = pd.read_csv("LSTM_submission.csv")
NB = pd.read_csv("NB_submission.csv")
ensemble_df = pd.read_csv("sample_submission.csv")

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

for i in np.arange(0.25, 1.0, 0.25):
    for label in list_classes:
        ensemble_df[label] = i*LSTM[label] + (1-i)*NB[label]
    print(ensemble_df.shape)
    ensemble_df.to_csv("ensemble_submission_%s.csv" % i, index=False)
