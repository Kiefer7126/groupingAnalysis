# -*- coding: utf-8 -*-

import numpy as np
import math
from copy import deepcopy
import GLCMfeatures

class Clustering:
    def __init__(self, audioduration, beatStructure, glcms):
        self.glcms = glcms
        self.features = []
        self.distances = []
        self.groupingStructure = {"audioduration":audioduration, "grouping":[beatStructure["beat"]]}

    def test(self, feature):

        glcmfeature = GLCMfeatures.GLCMfeatures()
        self.features = deepcopy(feature)
        self.calcDistance(self.features)

        for i in range(len(self.groupingStructure["grouping"][0])-1):

            prevGrouping = deepcopy(self.groupingStructure["grouping"][i])
            headIndex = self.distances.index(min(self.distances))

            print("-----headIndex-----")
            print(headIndex)
            print("-----grouping-----")
            print(prevGrouping)
            print("-----distances-----")
            print(self.distances)

            if(headIndex != (len(prevGrouping) -1)):
                prevGrouping.pop(headIndex + 1)
                np.delete(self.features, headIndex+1, 1)
                self.glcms[headIndex] = (self.glcms[headIndex] + self.glcms[headIndex + 1])/2
                self.glcms.pop(headIndex + 1)

            if (headIndex == (len(self.distances) - 1) ):
                self.distances.pop(headIndex)
            else:
                self.distances.pop(headIndex+1)

            glcmfeature.calcSumGLCMfeatures(self.glcms[headIndex])

            featureVector = [glcmfeature.contrast[i], glcmfeature.dissimilarity[i], glcmfeature.homogeneity[i], glcmfeature.ASM[i], glcmfeature.energy[i], glcmfeature.correlation[i]]
            print("-----featureVector-----")
            print(featureVector)

            self.features[:, headIndex] = featureVector
            self.groupingStructure["grouping"].append(prevGrouping)

            if(len(self.distances) != 0):
                self.distances[headIndex] = np.linalg.norm(self.features[:,headIndex] - feature[:, headIndex+1])

        print(self.groupingStructure)

    def calcDistance(self, feature):
        for i in range(feature.shape[1]-1):
            self.distances.append(np.linalg.norm(feature[:,i] - feature[:, i+1]))

        print("-----distance-----")
        print(self.distances)
        print("-----feature-----")
        print(feature)

        # for j in range(feature.shape[1]-1):
        #     for i in range(feature.shape[0]):
        #         distance += (feature[i, j] - feature[i, j+1])**2
        #     self.distances.append(math.sqrt(distance))
    def standardizationFeatures(self, features):
        features.contrast = self.standardization(features.contrast)
        features.dissimilarity = self.standardization(features.dissimilarity)
        features.homogeneity = self.standardization(features.homogeneity)
        features.ASM = self.standardization(features.ASM)
        features.energy = self.standardization(features.energy)
        #features.correlation = standardization(features.correlation) # 相関は標準化しなくて良い

    def standardization(self, vector):
        vector_copy = np.copy(vector)
        vector =  (vector_copy - vector_copy.mean()) / vector_copy.std()
        return vector
