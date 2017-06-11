# coding:UTF-8
import matplotlib.pyplot as pltssm
import librosa
import numpy as np

class SSM:
    def __init__(self, y, sr, features):
        self.y = y
        self.sr = sr
        self.features = features
        self.novelty = []

    def draw(self):
        ssm = librosa.segment.recurrence_matrix(self.features, metric='cosine', mode='affinity')

        np_novelty = self.makeNovelty(ssm)
        self.drawNovelty(self.novelty)

        #R2 = librosa.segment.recurrence_matrix(mfcc, mode='affinity', sym=True)
        lag = librosa.segment.recurrence_to_lag(ssm, pad=True)

        pltssm.figure(figsize=(8, 4))
        pltssm.subplot(1, 2, 1)
        librosa.display.specshow(ssm, x_axis='time', y_axis='time', aspect='equal', cmap = "gray_r")
        pltssm.title('Binary recurrence (symmetric)')

        pltssm.subplot(1, 2, 2)
        librosa.display.specshow(lag, x_axis='time', y_axis='lag', cmap = "gray_r")
        pltssm.title('Binary recurrence (symmetric)')

        #plt.tight_layout()
        pltssm.show()

        return np_novelty

    def makeNovelty(self, ssm):
        print(ssm)
        k = np.array([[-1,1],[1,-1]])
        a = np.array([1])#2*2用
        #a = np.array([[1,1],[1,1]])#4*4用

        kr = np.kron(k,a)

        print(kr)

# 2*2用
        for i in range(len(ssm)):
            print(i)
            print(len(ssm))
            if(i <= 1):
                n = kr[1-i:2,1-i:2] * ssm[0:i+1,0:i+1]
            else:
                n = kr * ssm[i-1:i+1,i-1:i+1]

            self.novelty.append(np.sum(n))

# 4*4用
        # for i in range(len(ssm)):
        #     print(i)
        #     print(len(ssm))
        #     if(i <= 2):
        #         n = kr[2-i:4,2-i:4] * ssm[0:i+2,0:i+2]
        #     elif(i > len(ssm)-2):
        #         n = kr[0:(len(ssm)-i+2),0:(len(ssm)-i+2)] * ssm[i-2:len(ssm),i-2:len(ssm)]
        #     else:
        #         n = kr * ssm[i-2:i+2,i-2:i+2]
        #
        #     self.novelty.append(np.sum(n))

# カーネルが中心じゃないやつ
        # for i in range(len(ssm)):
        #     if(i < (len(ssm)-3)):
        #         n = kr * ssm[i:i+4,i:i+4]
        #     else:
        #         n = kr[0:(len(ssm)-i),0:(len(ssm)-i)] * ssm[i:len(ssm),i:len(ssm)]
        #
        #     self.novelty.append(np.sum(n))

        np_novelty = np.array(self.novelty)

        np_novelty = np_novelty + abs(np.amin(np_novelty))
        np_novelty = np_novelty / np.amax(np_novelty)

        print("novelty")
        print(self.novelty)
        print(np_novelty)

        return np_novelty

    def drawNovelty(self,data):
        pltssm.figure(2)
        #pltssm.subplot(6, 1, index) # 2行1列の2番目
        pltssm.plot(data)
