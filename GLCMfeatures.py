# -*- coding: utf-8 -*-

import skimage
import math
from skimage import io
from skimage.feature import greycomatrix, greycoprops
import skimage.color as color
import numpy as np
import matplotlib.pyplot as pltglcm

class GLCMfeatures:
    def __init__(self):
        self.contrast = []
        self.dissimilarity = []
        self.homogeneity = []
        self.ASM = []
        self.energy = []
        self.correlation = []
        self.distances = []
        self.glcms = []

    def calcGLCMfeatures(self, glcm, distance, direction):
        self.contrast.append(greycoprops(glcm, 'contrast')[distance][direction]) #[d, a] d'th distance and a'th angle
        self.dissimilarity.append(greycoprops(glcm, 'dissimilarity')[distance][direction])
        self.homogeneity.append(greycoprops(glcm, 'homogeneity')[distance][direction])
        self.ASM.append(greycoprops(glcm, 'ASM')[distance][direction])
        self.energy.append(greycoprops(glcm, 'energy')[distance][direction])
        self.correlation.append(greycoprops(glcm, 'correlation')[distance][direction])

    def calcSumGLCMfeatures(self, glcm):
        self.contrast.append(self.calcContrast(glcm))
        self.dissimilarity.append(self.calcDissimilarity(glcm))
        self.homogeneity.append(self.calcHomogeneity(glcm))
        self.ASM.append(self.calcASM(glcm))
        self.energy.append(self.calcEnergy(glcm))
        self.correlation.append(self.calcCorrelation(glcm))

    def calcGLCM(self, binName, distance, direction):
        image = io.imread(binName)
        grayImage = color.rgb2gray(image)
        gray256Image = skimage.img_as_ubyte(grayImage)
        io.imshow(gray256Image)
        glcm = greycomatrix(gray256Image, distance, direction, levels=256, normed=True, symmetric=True)
        #print(glcm[:, :, 0, 0]) # [i, j, d, theta]

        return glcm

    def sumGLCM(self, glcm):
        nm_0_glcm = np.array(glcm[:,:,0,0])
        nm_45_glcm = np.array(glcm[:,:,0,1])
        nm_90_glcm = np.array(glcm[:,:,0,2])
        nm_135_glcm = np.array(glcm[:,:,0,3])
        nm_sum_glcm = nm_0_glcm + nm_45_glcm + nm_90_glcm + nm_135_glcm
        nm_sum_glcm = nm_sum_glcm / nm_sum_glcm.sum()

        return nm_sum_glcm

    # def calcGLCMfeaturesTest(self, glcm, distance, direction):

    def calcContrast(self, glcm):
        contrast = 0

        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                contrast += glcm[i,j] * (i - j) * (i - j)

        return contrast

    def calcDissimilarity(self, glcm):
        dissimilarity = 0

        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                dissimilarity += glcm[i,j] * abs(i - j)

        return dissimilarity

    def calcHomogeneity(self, glcm):
        homogeneity = 0

        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                homogeneity += glcm[i,j] / (1 + (i - j) * (i - j))

        return homogeneity

    def calcASM(self, glcm):
        asm = 0

        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                asm += glcm[i,j] * glcm[i,j]

        return asm

    def calcEnergy(self, glcm):
        asm = self.calcASM(glcm)
        energy = math.sqrt(asm)

        return energy

    def calcCorrelation(self, glcm):
        correlation = 0
        mx = 0
        my = 0
        sx = 0
        sy = 0

        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                mx += i * glcm[i,j]
                my += j * glcm[i,j]

        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                sx += (i - mx) * (i - mx) * glcm[i,j]
                sy += (j - my) * (j - my) * glcm[i,j]

        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                correlation += ((i - mx) * (j - my) * glcm[i,j] / math.sqrt(sx * sy))

        # for i in range(glcm.shape[0]):
        #     for j in range(glcm.shape[1]):
        #
        #         correlation += glcm[i,j] * ((i - glcm.mean(axis = 0)[i]) * (j - glcm.mean(axis = 1)[j]) / math.sqrt(glcm.std(axis = 0)[i] * glcm.std(axis = 0)[i] * glcm.std(axis = 1)[j] * glcm.std(axis = 1)[j] + 1))

        return correlation

    def calcGLCMfeaturesDistance(self):
        self.distances.append(self.calcDistance(self.contrast))
        self.distances.append(self.calcDistance(self.dissimilarity))
        self.distances.append(self.calcDistance(self.homogeneity))
        self.distances.append(self.calcDistance(self.ASM))
        self.distances.append(self.calcDistance(self.energy))
        self.distances.append(self.calcDistance(self.correlation))

    def calcDistance(self, data):
        distance = []
        for i in range(len(data)-1):
            aFeatures = np.array(data[i])
            bFeatures = np.array(data[i+1])
            distance.append(np.linalg.norm(aFeatures - bFeatures))
        return distance

    def drawGLCMfeatures(self):
        pltglcm.figure(2)
        #plt.title("GLCM Features", fontsize=25, fontname='serif')
        #plt.legend(('0 degree', '45 degree', '90 degree', '135 degree'), loc='400')
        self.drawGraph('contrast', self.contrast, 1)
        self.drawGraph('dissimilarity', self.dissimilarity, 2)
        self.drawGraph('homogeneity', self.homogeneity, 3)
        self.drawGraph('ASM', self.ASM, 4)
        #drawGraph('energy', agent.energy, 5)
        self.drawGraph('correlation', self.correlation, 5)

        pltglcm.figure(3)
        #plt.title("Features Distance")
        #plt.legend(('0 degree', '45 degree', '90 degree', '135 degree'))
        for i in range(len(self.distances)):
            self.drawGraph('distance', self.distances[i], i+1)

    def drawGraph(self, label, data, index):
        pltglcm.subplot(6, 1, index) # 2行1列の2番目
        pltglcm.plot(data)
        pltglcm.ylabel(label)

    def printFeatures(self):
        print(self.contrast)
        print(self.dissimilarity)
        print(self.homogeneity)
        print(self.ASM)
        print(self.correlation)
