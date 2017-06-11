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

    def makeGroupingStructure(self, feature, np_novelty):

        numberOfBin = len(self.groupingStructure["grouping"][0])
        numberOfmergedBins = []
        numberOfmergedBins = numberOfBin * [1]

        glcmfeature = GLCMfeatures.GLCMfeatures()
        #self.features = deepcopy(feature)
        self.features = np.array(feature)

        self.calcDistance(self.features)

        np_distances = np.array(self.distances)

        print("---------------len(np_distances)-------------")

        print(len(np_distances))


        k = np.ones((len(np_distances)+1) / 4)
        b_metric = np.array([1, 0.25, 0.5, 0.25])#linear
        #b_metric = np.array([1,  math.sqrt(0.25), math.sqrt(0.5),  math.sqrt(0.25)])#log
        metric = np.kron(k,b_metric)
        print(metric)
        #np_distances = np_novelty[:len(np_novelty)-1] * np_distances
        np_distances = np.delete(np_novelty,0) * np_distances * np.delete(metric, 0)
        self.distances = np_distances.tolist()
        list_metric = np.delete(metric,0).tolist()

        for i in range(len(self.groupingStructure["grouping"][0])-1):

            prevGrouping = deepcopy(self.groupingStructure["grouping"][i])
            headIndex = self.distances.index(min(self.distances))

            print("-----headIndex-----")
            print(headIndex)
            print("-----grouping-----")
            print(prevGrouping)
            print("-----distances-----")
            print(self.distances)
            print("-----metric-----")
            print(list_metric)
            print("-----features-----")
            print(self.features)

            if(headIndex != (len(prevGrouping) -1)):
                prevGrouping.pop(headIndex + 1)
                self.features = np.delete(self.features, headIndex+1, 1)
                self.glcms[headIndex] = (self.glcms[headIndex] + self.glcms[headIndex + 1])/2
                self.glcms.pop(headIndex + 1)

            if (headIndex == (len(self.distances) - 1) ):
                self.distances.pop(headIndex)
                list_metric.pop(headIndex)
            else:
                self.distances.pop(headIndex+1)
                list_metric.pop(headIndex)

            glcmfeature.calcSumGLCMfeatures(self.glcms[headIndex])

            featureVector = [glcmfeature.contrast[i], glcmfeature.dissimilarity[i], glcmfeature.homogeneity[i], glcmfeature.ASM[i], glcmfeature.energy[i], glcmfeature.correlation[i]]
            print("-----featureVector-----")
            print(featureVector)

            self.features[:, headIndex] = featureVector
            self.groupingStructure["grouping"].append(prevGrouping)

            if(len(self.distances) != 0):
                print("features")
                print(self.features)
                print(len(self.features[0]))

                numberOfmergedBins[headIndex] += numberOfmergedBins[headIndex+1]
                numberOfmergedBins.pop(headIndex+1)

                if(len(self.features[0]) > headIndex+1):
                    self.distances[headIndex] = np.linalg.norm(self.features[:,headIndex] - feature[:, headIndex+1])
                    self.distances[headIndex] = self.distances[headIndex] * numberOfmergedBins[headIndex] * numberOfmergedBins[headIndex+1]* list_metric[headIndex]#両側
                    #self.distances[headIndex] = self.distances[headIndex] * numberOfmergedBins[headIndex] #片側
                if(headIndex != 0):
                    self.distances[headIndex-1] = np.linalg.norm(self.features[:,headIndex-1] - feature[:, headIndex])
                    self.distances[headIndex-1] = self.distances[headIndex-1] * numberOfmergedBins[headIndex] * numberOfmergedBins[headIndex-1]*list_metric[headIndex-1]#両側
                    #self.distances[headIndex-1] = self.distances[headIndex-1] * numberOfmergedBins[headIndex] #片側

            print("numberOfmergedbins")
            print(numberOfmergedBins)

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
