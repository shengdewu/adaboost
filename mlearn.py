import numpy as np

class adaboost(object):
    def loadSimpData(self):
        datMat = np.matrix([[1., 2.1],
                         [2., 1.1],
                         [1.3, 1.],
                         [1., 1.],
                         [2., 1.]])
        classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
        return datMat, classLabels

    def train(self, dataMat, dataLabel, iterNum=20):
        sampleNum, featNum= np.mat(dataMat).shape
        weight = np.mat(np.ones((sampleNum,1))/sampleNum)
        precClass = []
        for iter in range(iterNum):
            stump = self.buildStump(dataMat, dataLabel,weight)
            alph = 0.5 * np.log((1-stump['bestErr'])/max(stump['bestErr'], 1e-16))
            stump['alph'] = alph #弱分类器权重
            expon = np.exp(np.multiply(-1 * alph * stump['bestClass'], np.mat(dataLabel).T))
            z = np.multiply(expon, weight)
            weight = z/np.sum(z)
            precClass.append(stump)
        return precClass

    def buildStump(self, dataTrain, dataLabel, weight):
        '''
        遍历每一个纬度，计算最小分类误差
        '''
        dataMat = np.matrix(dataTrain)
        labelMat = np.matrix(dataLabel)
        sampleNum, featNum = dataMat.shape

        bestErr = np.inf
        stump = {}
        for c_dim in range(featNum): #遍历每个特征
            max = dataMat[:,c_dim].max()
            min = dataMat[:,c_dim].min()
            stepSize = float(max - min) / sampleNum
            for step in range(1, sampleNum-1):
                for method in ['lt', 'gt']:
                    thres = float(min) + float(step) * stepSize
                    predictClass = self.__caclClass__(dataMat, c_dim, thres, method)
                    errArr = np.mat(np.ones((5, 1))) #numpy 的array和mat 差飞了 不要混合运算
                    errArr[predictClass == labelMat.T] = 0
                    err = errArr.T*weight
                    if bestErr > err[0,0]:
                        bestErr = err[0,0]
                        stump['bestErr'] = bestErr
                        stump['bestDim'] = c_dim
                        stump['bestMethod'] = method
                        stump['bestThres'] = thres
                        stump['bestClass'] = predictClass.copy()
        return stump

    def __caclClass__(self, dataMat, dim, thres, method):
        classLabel = np.ones((dataMat.shape[0], 1))
        if method == 'lt':
            classLabel[dataMat[:,dim] <= thres] = -1
        elif method == 'gt':
            classLabel[dataMat[:,dim] > thres] = -1

        return classLabel
