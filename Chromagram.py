# coding:UTF-8
import matplotlib.pyplot as pltchroma
import librosa
import numpy as np

class Chromagram:
    def __init__(self, y, sr, filename):
        self.y = y
        self.sr = sr
        self.filename = filename

    def calc(self):
        self.chroma = librosa.feature.chroma_stft(y=self.y, sr=self.sr, norm=0)
        return self.chroma

    def draw(self):
        D = np.abs(librosa.stft(self.y))**2
        pltchroma.figure(figsize=(8, 4))
        pltchroma.subplot(2, 1, 1)
        librosa.display.specshow(librosa.logamplitude(D, ref_power=np.max), x_axis='time', y_axis='linear', cmap = "Spectral")
        pltchroma.title('Power spectrogram')

        pltchroma.subplot(2, 1, 2)

        # plt.figure(figsize=(10, 4))
        librosa.display.specshow(self.chroma, y_axis='chroma', x_axis='time', cmap = "Spectral")

        pltchroma.show()

    def export(self, norm):
        chroma = librosa.feature.chroma_stft(y=self.y, sr=self.sr, norm = norm)
        pltchroma.figure(figsize=(len(chroma[0])/100.0, len(chroma)/100.0))
        librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', cmap = "gray_r")
        pltchroma.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        pltchroma.axis('off')
        pltchroma.savefig(self.filename + "_chromagram.png")
