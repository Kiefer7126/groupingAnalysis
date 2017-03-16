# coding:UTF-8
import matplotlib.pyplot as plt
import librosa
import numpy as np

class Onset:
    def __init__(self, y, sr):
        self.y = y
        self.sr = sr

    def detection(self):
        self.onset_frames = librosa.onset.onset_detect(y=self.y, sr=self.sr)
        self.onset_times = librosa.frames_to_time(self.onset_frames[:], sr=self.sr)
        onset_times_int = self.onset_times * 100 # 有効数字3桁の整数
        onset_times_int = onset_times_int.astype(np.int64)

        return onset_times_int

    def draw(self):
        D = np.abs(librosa.stft(self.y))**2
        plt.figure(figsize=(8, 4))
        plt.subplot(2, 1, 1)
        librosa.display.specshow(librosa.logamplitude(D, ref_power=np.max), x_axis='time', y_axis='linear', cmap = "Spectral")
        plt.title('Power spectrogram')
        plt.subplot(2, 1, 2)

        self.o_env = librosa.onset.onset_strength(self.y, sr=self.sr)
        times = librosa.frames_to_time(np.arange(len(self.o_env)), sr=self.sr)
        #plt.plot(o_env, label='Onset strength')

        # vlines(x, ymin, ymax, colors='k', linestyles='solid', label='', hold=None, data=None, **kwargs)
        plt.vlines(times, 0, self.o_env, color='g', alpha=0.9, linestyle='-', label='Strength')

        plt.vlines(self.onset_times, 0, max(self.o_env), color='r', alpha=0.9, linestyle='--', label='Onsets')

        plt.xticks([])
        plt.axis('tight')
        plt.legend(frameon=True, framealpha=0.75)

        plt.show()
