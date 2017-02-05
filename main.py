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

windowSize = 1024
siftSize = windowSize / 4

class glcmFeatures:
    def __init__(self):
        self.contrast = []
        self.dissimilarity = []
        self.homogeneity = []
        self.ASM = []
        self.energy = []
        self.correlation = []
        self.distances = []

glcmFeatureNames = ["contrast","dissimilarity","homogeneity","ASM","energy", "correlation"]
direction = [0, np.pi/4, np.pi/2, 3 * np.pi/4]

#                 0          1        2         3          4
methodList = ["complete", "ward", "average", "single", "centroid"]
method = methodList[2]

#                 0           1           2        3         4
filenameGPR2 = ["GPR2","GPR2-inverse","GPR2-a","GPR2-b","GPR2-slow"]

#                 0           1           2        3       4        5
filenameGPR3 = ["GPR3","GPR3-inverse","GPR3-a","GPR3-b","GPR3-c","GPR3-d"]

#                 0     1          2              3            4             5           6          7            8                9            10          11           12           13            14
filenameList = ["002","038","C_clean-dist","C_dist-clean","fred_clean", "fred_dist","k550-120","k550-120-2","k550-120-teisei","k550-120-4","k550-180","star_clean","star_dist", filenameGPR2, filenameGPR3]

def main():
    filename = "testData/" + filenameList[14][4]
    y, sr = librosa.load(filename + ".wav")
    drawSpectrogram(y, sr, filename)
    print(sr)

    #-----拍で分割する場合に使用する-----
    beatStructure = loadBeat(filename)
    divideSpectrograms = divideSpectrogramByBeat(filename, beatStructure, sr)
    #divideSpectrograms = divideSpectrogramBy8Beat(filename, beatStructure, sr)
    #----------------------------------

    #divideSpectrograms = divideSpectrogram(filename)

    files = os.listdir(filename)
    agent0 = glcmFeatures()
    agent45 = glcmFeatures()
    agent90 = glcmFeatures()
    agent135 = glcmFeatures()

    for binpng in files:
        binName = os.path.join(filename +"/" + binpng)
        glcm = calcGLCM(binName)
        calcGLCMfeatures(agent0, glcm, 0)
        calcGLCMfeatures(agent45, glcm, 1)
        calcGLCMfeatures(agent90, glcm, 2)
        calcGLCMfeatures(agent135, glcm, 3)

    standardizationFeatures(agent0)
    standardizationFeatures(agent45)
    standardizationFeatures(agent90)
    standardizationFeatures(agent135)

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
    drawGLCMfeatures(agent90)
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
    glcm = greycomatrix(gray256Image, [1], direction, levels=256, normed=True, symmetric=True)
    #print(glcm[:, :, 0, 0]) # [i, j, d, theta]
    return glcm

def calcGLCMfeaturesDistance(features):
    features.distances.append(calcDistance(features.contrast))
    features.distances.append(calcDistance(features.dissimilarity))
    features.distances.append(calcDistance(features.homogeneity))
    features.distances.append(calcDistance(features.ASM))
    features.distances.append(calcDistance(features.energy))
    features.distances.append(calcDistance(features.correlation))

def calcGLCMfeatures(agent, glcm, direction):
    agent.contrast.append(greycoprops(glcm, 'contrast')[0][direction]) #[d, a] d'th distance and a'th angle
    agent.dissimilarity.append(greycoprops(glcm, 'dissimilarity')[0][direction])
    agent.homogeneity.append(greycoprops(glcm, 'homogeneity')[0][direction])
    agent.ASM.append(greycoprops(glcm, 'ASM')[0][direction])
    agent.energy.append(greycoprops(glcm, 'energy')[0][direction])
    agent.correlation.append(greycoprops(glcm, 'correlation')[0][direction])

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

def divideSpectrogramByBeat(filename, beatStructure, sr):
    spectrogram = Image.open(filename + '_spectrogram.png')
    invSr = 1.0/sr
    samplePsec = invSr * siftSize
    divideSpectrograms = []
    beatSize = len(beatStructure["beat"])

    if os.path.isdir(filename):
        print ("ok")
    else:
        os.mkdir(filename)

    for i in range(len(beatStructure["beat"])-1):
        divSpectrogram = spectrogram.crop((beatStructure["beat"][i] / 100.0 / samplePsec, 0, beatStructure["beat"][i+1] / 100.0 / samplePsec, windowSize/2)) #(left, top, right, bottom)
        divideSpectrograms.append(divSpectrogram)
        divSpectrogram.save(filename +'/'+ '{0:03d}'.format(i) + '.png')

    divSpectrogram = spectrogram.crop((beatStructure["beat"][beatSize-1] / 100.0 / samplePsec, 0, spectrogram.size[0], windowSize/2)) #(left, top, right, bottom)
    divideSpectrograms.append(divSpectrogram)
    divSpectrogram.save(filename +'/'+ '{0:03d}'.format(beatSize-1) + '.png')

    return divideSpectrograms

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

def divideSpectrogram(filename):
    spectrogram = Image.open(filename + '_spectrogram.png')
    divideSpectrograms = []
    splitSize = 15

    if os.path.isdir(filename):
        print ("ok")
    else:
        os.mkdir(filename)

    for i in range(spectrogram.size[0] / splitSize):
        divSpectrogram = spectrogram.crop((splitSize * i, 0, splitSize * (i+1), windowSize/2)) #(left, top, right, bottom)
        divideSpectrograms.append(divSpectrogram)
        divSpectrogram.save(filename +'/'+ '{0:03d}'.format(i) + '.png')

    return divideSpectrograms

def drawSpectrogram(y, sr, filename):
    S = np.abs(librosa.stft(y, n_fft=windowSize, hop_length=siftSize))
    plt.figure(figsize=(len(S[0])/100.0, len(S)/100.0))
    librosa.display.specshow(librosa.logamplitude(S**2, ref_power=np.median), sr=sr, y_axis='log', x_axis='time',cmap = "gray_r")
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.axis('off')
    plt.savefig(filename + "_spectrogram.png")

def loadBeat(filename):
    f = open(filename + '_beat.txt', 'r')
    return json.load(f)
    #keyList = beatStructure.keys()

if __name__ == '__main__':
    main()
