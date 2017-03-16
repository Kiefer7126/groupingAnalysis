# coding:UTF-8
import matplotlib.pyplot as plt
import librosa
import numpy as np

class SSM:
    def __init__(self, y, sr, features):
        self.y = y
        self.sr = sr
        self.features = features

    def draw(self):
        ssm = librosa.segment.recurrence_matrix(self.features, metric='cosine', mode='affinity')
        #R2 = librosa.segment.recurrence_matrix(mfcc, mode='affinity', sym=True)
        lag = librosa.segment.recurrence_to_lag(ssm, pad=True)

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        librosa.display.specshow(ssm, x_axis='time', y_axis='time', aspect='equal', cmap = "gray_r")
        plt.title('Binary recurrence (symmetric)')

        plt.subplot(1, 2, 2)
        librosa.display.specshow(lag, x_axis='time', y_axis='lag', cmap = "gray_r")
        plt.title('Binary recurrence (symmetric)')

        plt.tight_layout()
        plt.show()
