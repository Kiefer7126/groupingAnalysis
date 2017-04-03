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
import GLCMfeatures
from copy import deepcopy

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

distanceAgent = []

#                 0      1              2               3               4              5            6           7             8                  9             10         11          12          13           14            15             16    17     18        19                20        21
filenameGPR2 = ["GPR2", "GPR2-inverse", "GPR2-a"      , "GPR2-b"      , "GPR2-slow"]
filenameGPR3 = ["GPR3", "GPR3-inverse", "GPR3-a"      , "GPR3-b"      , "GPR3-c"    , "GPR3-d"]
filenameList = ["002" , "038"         , "C_clean-dist", "C_dist-clean", "fred_clean", "fred_dist", "k550-120", "k550-120-2", "k550-120-teisei", "k550-120-4", "k550-180","k550-orc", "star_clean","star_dist", filenameGPR2, filenameGPR3, "up", "up8", "octave", "piano and flute", "001", "001-mono"]
testList =     ["001" , "002-2"       , "009"         , "014"         , "038-2"     , "eien"]


def main():
    filename = "testData/" + "k550-r"
    y, sr = librosa.load(filename + ".wav")

    audioduration = float(len(y))/sr

    spectrogram = Spectrogram.Spectrogram(y, sr, windowSize, siftSize, filename)
    spectrogram.export(freqAxisType)

    onset = Onset.Onset(y, sr)

    onsetStructure = {}
    onsetStructure["audioduration"] = audioduration
    onsetList = onset.detection().tolist()
    onsetStructure["beat"] = onsetList
    exportOnset(filename, onsetStructure)

    #onset.draw()

    chromagram = Chromagram.Chromagram(y, sr, filename)
    chroma = chromagram.calc()
    #chromagram.draw()
    chromagram.export(np.inf)
    chromagram.export(None)

    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    ssm = SSM.SSM(y, sr, chroma)
    ssm.draw()

    #-----拍で分割する場合に使用する-----
    #beatStructure = loadBeat(filename)
    beatStructure = loadOnset(filename)
    print(beatStructure)

    divideSpectrograms = spectrogram.divideByBeat(beatStructure)

    #divideSpectrograms = divideSpectrogramBy8Beat(filename, beatStructure, sr)
    #----------------------------------

    #divideSpectrograms = spectrogram.divide(15)

    files = os.listdir(filename)

    agent0   = GLCMfeatures.GLCMfeatures()
    agent45  = GLCMfeatures.GLCMfeatures()

    # 距離ごとにGLCMを作成しようとしたときの残骸

    # for i in range(len(distance)):
    #     agent90 = glcmFeatures()
    #     distanceAgent.append(agent90)

    agent90  = GLCMfeatures.GLCMfeatures()
    agent135 = GLCMfeatures.GLCMfeatures()
    agentSUM = GLCMfeatures.GLCMfeatures()

    for binpng in files:
        binName = os.path.join(filename +"/" + binpng)
        glcm = agent0.calcGLCM(binName, distance, direction) # GLCMはどのエージェントでも同じ（わかりにくい）
        sumglcm = agent0.sumGLCM(glcm)

        # for i in range(len(distanceAgent)):
        #     calcGLCMfeatures(distanceAgent[i],  glcm, i, 2)

        agent0.calcGLCMfeatures(glcm, 0, 0)
        agent45.calcGLCMfeatures(glcm, 0, 1)
        agent90.calcGLCMfeatures(glcm, 0, 2)
        agent135.calcGLCMfeatures(glcm, 0, 3)

        agentSUM.calcSumGLCMfeatures(sumglcm)

    standardizationFeatures(agent0)
    standardizationFeatures(agent45)
    standardizationFeatures(agent90)
    standardizationFeatures(agent135)
    standardizationFeatures(agentSUM)

    # for i in range(len(distanceAgent)):
    #     standardizationFeatures(distanceAgent[i])

    agent0.calcGLCMfeaturesDistance()
    agent45.calcGLCMfeaturesDistance()
    agent90.calcGLCMfeaturesDistance()
    agent135.calcGLCMfeaturesDistance()
    agentSUM.calcGLCMfeaturesDistance()

    plt.clf()
    plt.figure(1)
    #plt.subplots_adjust(left=0.3, bottom=0, right=0.9, top=1, wspace=0.4, hspace=0.5)
    img = np.array( Image.open(filename + '_spectrogram.png') )
    #plt.subplot(7,1,1) # 7行1列の1番目
    plt.imshow(img)

    agent0.drawGLCMfeatures()
    agent45.drawGLCMfeatures()
    agent90.drawGLCMfeatures()
    agent135.drawGLCMfeatures()
    agentSUM.drawGLCMfeatures()

    drawDendrogram(agent0)
    drawDendrogram(agent45)
    drawDendrogram(agent90)
    drawDendrogram(agent135)
    result = drawDendrogram(agentSUM)

    # for i in range(len(distanceAgent)):
    #     drawGLCMfeatures(distanceAgent[i])

    plt.show()

    groupingStructure = {}
    groupingStructure["audioduration"] = audioduration
    groupingStructure["grouping"] = []
    groupingStructure["grouping"].append(beatStructure["beat"])

    checkList = range(0, len(beatStructure["beat"]))
    print(checkList)

    prev = 0

    for i in range(0, len(result)):
        prevGrouping = deepcopy(groupingStructure["grouping"][i])
        prevGrouping.remove(groupingStructure["grouping"][0][checkList[int(result[i][1])]])
        checkList.append(checkList[int(result[i][0])])
        groupingStructure["grouping"].append(prevGrouping)

    print(groupingStructure)

    exportGroup(filename, groupingStructure)

def drawDendrogram(agent):
    plt.figure()
    vectors = convStructToVector(agent)
    result = linkage(vectors, metric = 'euclidean', method = method)
    dendrogram(result, count_sort  = 'ascending')
    return result

def standardizationFeatures(features):
    features.contrast = standardization(features.contrast)
    features.dissimilarity = standardization(features.dissimilarity)
    features.homogeneity = standardization(features.homogeneity)
    features.ASM = standardization(features.ASM)
    features.energy = standardization(features.energy)
    #features.correlation = standardization(features.correlation) # 相関は標準化しなくて良い

def standardization(vector):
    vector_copy = np.copy(vector)
    vector =  (vector_copy - vector_copy.mean()) / vector_copy.std()
    return vector

def convStructToVector(features):
    vectors = []
    for i in range(len(features.contrast)):
        vector = []
        #vector = [features.contrast[i], features.dissimilarity[i], features.homogeneity[i], features.ASM[i], features.energy[i], features.correlation[i], i*1000]
        vector = [features.contrast[i], features.dissimilarity[i], features.homogeneity[i], features.ASM[i], features.correlation[i], i*1000]
        vectors.append(vector)
    return vectors

def drawBarGraph(label, data, index):
    plt.subplot(6, 1, index) # 6行1列のi番目
    plt.bar(range(len(data)), data, width=0.3)
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

def loadOnset(filename):
    f = open(filename + '_onset.txt', 'r')
    return json.load(f)
    #keyList = beatStructure.keys()

def exportOnset(filename, structure):
    f = open(filename + "_onset.txt", "w")
    json.dump(structure, f)

def exportGroup(filename, structure):
    f = open(filename + "_group_Predict.txt", "w")
    json.dump(structure, f)

if __name__ == '__main__':
    main()
