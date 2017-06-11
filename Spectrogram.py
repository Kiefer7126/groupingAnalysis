# coding:UTF-8
import matplotlib.pyplot as pltspect
import librosa
import numpy as np
from PIL import Image
import os

class Spectrogram:
    def __init__(self, y, sr, windowSize, siftSize, filename):
        self.y = y
        self.sr = sr
        self.windowSize = windowSize
        self.siftSize = siftSize
        self.filename = filename

    def export(self, freqAxisType):
        S = np.abs(librosa.stft(self.y, n_fft=self.windowSize, hop_length=self.siftSize))
        pltspect.figure(figsize=(len(S[0])/100.0, len(S)/100.0))
        librosa.display.specshow(librosa.logamplitude(S**2, ref_power=np.median), sr=self.sr, y_axis = freqAxisType, x_axis='time',cmap = "gray_r")
        pltspect.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        pltspect.axis('off')
        pltspect.savefig(self.filename + "_spectrogram.png")

    def divide(self, splitSize):
        spectrogram = Image.open(self.filename + '_spectrogram.png')
        divideSpectrograms = []

        if os.path.isdir(self.filename):
            print ("ok")
        else:
            os.mkdir(self.filename)

        for i in range(spectrogram.size[0] / splitSize):
            divSpectrogram = spectrogram.crop((splitSize * i, 0, splitSize * (i+1), self.windowSize/2)) #(left, top, right, bottom)
            divideSpectrograms.append(divSpectrogram)
            divSpectrogram.save(self.filename +'/'+ '{0:03d}'.format(i) + '.png')

        return divideSpectrograms


    def divideByBeat(self, beatStructure):
        spectrogram = Image.open(self.filename + '_spectrogram.png')
        invSr = 1.0/self.sr
        samplePsec = invSr * self.siftSize
        divideSpectrograms = []
        beatSize = len(beatStructure["beat"])

        if os.path.isdir(self.filename):
            print ("ok")
        else:
            os.mkdir(self.filename)

        for i in range(len(beatStructure["beat"])-1):
            divSpectrogram = spectrogram.crop((beatStructure["beat"][i] / 100.0 / samplePsec, 0, beatStructure["beat"][i+1] / 100.0 / samplePsec, self.windowSize/2)) #(left, top, right, bottom)
            divideSpectrograms.append(divSpectrogram)
            divSpectrogram.save(self.filename +'/'+ '{0:03d}'.format(i) + '.png')

        divSpectrogram = spectrogram.crop((beatStructure["beat"][beatSize-1] / 100.0 / samplePsec, 0, spectrogram.size[0], self.windowSize/2)) #(left, top, right, bottom)
        divideSpectrograms.append(divSpectrogram)
        divSpectrogram.save(self.filename +'/'+ '{0:03d}'.format(beatSize-1) + '.png')

        return divideSpectrograms
