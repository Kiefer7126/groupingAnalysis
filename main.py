# -*- coding: utf-8 -*-

from __future__ import print_function
import librosa
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
import os
import skimage
from skimage import io
from skimage.feature import greycomatrix, greycoprops
import skimage.color as color
import scipy
from scipy.cluster.hierarchy import linkage, dendrogram

import Onset
import Chromagram
import SSM
import Spectrogram

# ---------- パラメータ ----------
windowSize = 1024
siftSize = windowSize / 4

freqAxisList = ['linear', 'log', 'mel']
freqAxisType = freqAxisList[0]

glcmFeatureNames = ["contrast","dissimilarity","homogeneity","ASM","energy", "correlation"]
direction = [0, np.pi/4, np.pi/2, 3 * np.pi/4]

# distance = [x for x in range(30)]
distance = [1]

methodList = ["complete", "ward", "average", "single", "centroid"]
method = methodList[0]
# -------------------------------

class glcmFeatures:
    def __init__(self):
        self.contrast = []
        self.dissimilarity = []
        self.homogeneity = []
        self.ASM = []
        self.energy = []
        self.correlation = []
        self.distances = []

distanceAgent = []

#                 0      1              2               3               4              5            6           7             8                  9             10         11          12          13           14            15             16    17     18        19                20        21
filenameGPR2 = ["GPR2", "GPR2-inverse", "GPR2-a"      , "GPR2-b"      , "GPR2-slow"]
filenameGPR3 = ["GPR3", "GPR3-inverse", "GPR3-a"      , "GPR3-b"      , "GPR3-c"    , "GPR3-d"]
filenameList = ["002" , "038"         , "C_clean-dist", "C_dist-clean", "fred_clean", "fred_dist", "k550-120", "k550-120-2", "k550-120-teisei", "k550-120-4", "k550-180","k550-orc", "star_clean","star_dist", filenameGPR2, filenameGPR3, "up", "up8", "octave", "piano and flute", "001", "001-mono"]
testList =     ["001" , "002-2"       , "009"         , "014"         , "038-2"     , "eien"]


def main():
    filename = "testData/" + "001"
    y, sr = librosa.load(filename + ".wav", sr = 44100)

    spectrogram = Spectrogram.Spectrogram(y, sr, windowSize, siftSize, filename)
    spectrogram.export(freqAxisType)

    onset = Onset.Onset(y, sr)
    print(onset.detection())
    onset.draw()

    chromagram = Chromagram.Chromagram(y, sr, filename)
    chroma = chromagram.calc()
    chromagram.draw()
    chromagram.export()

    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    ssm = SSM.SSM(y, sr, chroma)
    ssm.draw()

    #-----拍で分割する場合に使用する-----
    beatStructure = loadBeat(filename)
    print(beatStructure)

    divideSpectrograms = spectrogram.divideByBeat(beatStructure)

    #divideSpectrograms = divideSpectrogramBy8Beat(filename, beatStructure, sr)
    #----------------------------------

    #divideSpectrograms = spectrogram.divide(15)

    files = os.listdir(filename)
    agent0   = glcmFeatures()
    agent45  = glcmFeatures()

    for i in range(len(distance)):
        agent90 = glcmFeatures()
        distanceAgent.append(agent90)

    # agent90  = glcmFeatures()
    agent135 = glcmFeatures()

    for binpng in files:
        binName = os.path.join(filename +"/" + binpng)
        glcm = calcGLCM(binName)

        calcGLCMfeatures(agent0,   glcm, 0, 0)
        calcGLCMfeatures(agent45,  glcm, 0, 1)

        for i in range(len(distanceAgent)):
            calcGLCMfeatures(distanceAgent[i],  glcm, i, 2)

        calcGLCMfeatures(agent135, glcm, 0, 3)

    for i in range(len(distanceAgent)):
        print(distanceAgent[i].contrast)

    standardizationFeatures(agent0)
    standardizationFeatures(agent45)

    # for i in range(len(distanceAgent)):
    #     standardizationFeatures(distanceAgent[i])

    standardizationFeatures(agent135)

    for i in range(len(distanceAgent)):
        print(distanceAgent[i].contrast)

    calcGLCMfeaturesDistance(agent0)
    calcGLCMfeaturesDistance(agent45)
    calcGLCMfeaturesDistance(agent90)
    calcGLCMfeaturesDistance(agent135)

    plt.clf()
    plt.figure(1)
    #plt.subplots_adjust(left=0.3, bottom=0, right=0.9, top=1, wspace=0.4, hspace=0.5)
    img = np.array( Image.open(filename + '_spectrogram.png') )
    #plt.subplot(7,1,1) # 7行1列の1番目
    plt.imshow(img)

    # drawGLCMfeatures(agent0)
    # drawDendrogram(agent0)
    # drawGLCMfeatures(agent45)
    # drawDendrogram(agent45)

    for i in range(len(distanceAgent)):
        drawGLCMfeatures(distanceAgent[i])

    drawDendrogram(agent90)
    # drawGLCMfeatures(agent135)
    # drawDendrogram(agent135)

    plt.show()

def drawGLCMfeatures(agent):
    plt.figure(2)
    #plt.title("GLCM Features", fontsize=25, fontname='serif')
    #plt.legend(('0 degree', '45 degree', '90 degree', '135 degree'), loc='400')
    drawGraph('contrast', agent.contrast, 1)
    drawGraph('dissimilarity', agent.dissimilarity, 2)
    drawGraph('homogeneity', agent.homogeneity, 3)
    drawGraph('ASM', agent.ASM, 4)
    #drawGraph('energy', agent.energy, 5)
    drawGraph('correlation', agent.correlation, 5)

    plt.figure(3)
    #plt.title("Features Distance")
    #plt.legend(('0 degree', '45 degree', '90 degree', '135 degree'))
    for i in range(len(agent.distances)):
        drawGraph('distance', agent.distances[i], i+1)

def drawDendrogram(agent):
    plt.figure()
    vectors = convStructToVector(agent)
    result = linkage(vectors, metric = 'euclidean', method = method)
    dendrogram(result, count_sort  = 'ascending')

def standardizationFeatures(features):
    features.contrast = standardization(features.contrast)
    features.dissimilarity = standardization(features.dissimilarity)
    features.homogeneity = standardization(features.homogeneity)
    features.ASM = standardization(features.ASM)
    features.energy = standardization(features.energy)
    features.correlation = standardization(features.correlation)

def standardization(vector):
    vector_copy = np.copy(vector)
    vector =  (vector_copy - vector_copy.mean()) / vector_copy.std()
    return vector

def convStructToVector(features):
    vectors = []
    for i in range(len(features.contrast)):
        vector = []
        #vector = [features.contrast[i], features.dissimilarity[i], features.homogeneity[i], features.ASM[i], features.energy[i], features.correlation[i], i*1000]
        vector = [features.contrast[i], features.dissimilarity[i], features.homogeneity[i], features.ASM[i], features.correlation[i], i*10]
        vectors.append(vector)
    return vectors

def calcGLCM(binName):
    image = io.imread(binName)
    grayImage = color.rgb2gray(image)
    gray256Image = skimage.img_as_ubyte(grayImage)
    io.imshow(gray256Image)
    glcm = greycomatrix(gray256Image, distance, direction, levels=256, normed=True, symmetric=True)
    #print(glcm[:, :, 0, 0]) # [i, j, d, theta]
    return glcm

def calcGLCMfeaturesDistance(features):
    features.distances.append(calcDistance(features.contrast))
    features.distances.append(calcDistance(features.dissimilarity))
    features.distances.append(calcDistance(features.homogeneity))
    features.distances.append(calcDistance(features.ASM))
    features.distances.append(calcDistance(features.energy))
    features.distances.append(calcDistance(features.correlation))

def calcGLCMfeatures(agent, glcm, distance, direction):
    agent.contrast.append(greycoprops(glcm, 'contrast')[distance][direction]) #[d, a] d'th distance and a'th angle
    agent.dissimilarity.append(greycoprops(glcm, 'dissimilarity')[distance][direction])
    agent.homogeneity.append(greycoprops(glcm, 'homogeneity')[distance][direction])
    agent.ASM.append(greycoprops(glcm, 'ASM')[distance][direction])
    agent.energy.append(greycoprops(glcm, 'energy')[distance][direction])
    agent.correlation.append(greycoprops(glcm, 'correlation')[distance][direction])

def calcDistance(data):
    distance = []
    for i in range(len(data)-1):
        aFeatures = np.array(data[i])
        bFeatures = np.array(data[i+1])
        distance.append(np.linalg.norm(aFeatures - bFeatures))
    return distance

def drawBarGraph(label, data, index):
    plt.subplot(6, 1, index) # 6行1列のi番目
    plt.bar(range(len(data)), data, width=0.3)
    plt.ylabel(label)

def drawGraph(label, data, index):
    plt.subplot(6, 1, index) # 2行1列の2番目
    plt.plot(data)
    plt.ylabel(label)

def divideSpectrogramBy8Beat(filename, beatStructure, sr):
    spectrogram = Image.open(filename + '_spectrogram.png')
    invSr = 1.0/sr
    samplePsec = invSr * siftSize
    divideSpectrograms = []
    beatSize = len(beatStructure["beat"])

    if os.path.isdir(filename):
        print ("ok")
    else:
        os.mkdir(filename)

    for i in range(beatSize-1):

        PrimPosition = beatStructure["beat"][i] / 100.0 / samplePsec
        NextPosition = beatStructure["beat"][i+1] / 100.0 / samplePsec
        shiftLength = (NextPosition - PrimPosition) / 2
        divSpectrogram = spectrogram.crop((PrimPosition, 0, NextPosition, windowSize/2)) #(left, top, right, bottom)
        divideSpectrograms.append(divSpectrogram)
        divSpectrogram.save(filename +'/'+ '{0:03d}'.format(2*i) + '.png')

        divSpectrogram = spectrogram.crop((PrimPosition + shiftLength, 0, NextPosition + shiftLength, windowSize/2)) #(left, top, right, bottom)
        divideSpectrograms.append(divSpectrogram)
        divSpectrogram.save(filename +'/'+ '{0:03d}'.format(2*i+1) + '.png')

    divSpectrogram = spectrogram.crop((beatStructure["beat"][beatSize-1] / 100.0 / samplePsec, 0, spectrogram.size[0], windowSize/2)) #(left, top, right, bottom)
    divideSpectrograms.append(divSpectrogram)
    divSpectrogram.save(filename +'/'+ '{0:03d}'.format(2 * (beatSize-1)) + '.png')

    divSpectrogram = spectrogram.crop((beatStructure["beat"][beatSize-1] / 100.0 / samplePsec + shiftLength, 0, spectrogram.size[0], windowSize/2)) #(left, top, right, bottom)
    divideSpectrograms.append(divSpectrogram)
    divSpectrogram.save(filename +'/'+ '{0:03d}'.format(2 * (beatSize-1) + 1) + '.png')

    return divideSpectrograms

def loadBeat(filename):
    f = open(filename + '_beat.txt', 'r')
    return json.load(f)
    #keyList = beatStructure.keys()

if __name__ == '__main__':
    main()
