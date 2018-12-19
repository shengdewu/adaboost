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
        aggClass =np.mat(np.zeros((sampleNum, 1)))
        for iter in range(iterNum):
            bestStump = self.buildStump(dataMat, dataLabel,weight)
            alph = float(0.5 * np.log((1-bestStump['err'])/max(bestStump['err'], 1e-16)))
            bestStump['alph'] = alph #弱分类器权重
            precClass.append(bestStump)
            expon = np.exp(np.multiply(-1 * alph * bestStump['class'], np.mat(dataLabel).T))
            z = np.multiply(expon, weight)
            weight = z/np.sum(z)
            aggClass += alph * bestStump['class']
            print(str('weight {} \n bestClass{}\n aggClass').format(weight.T, bestStump['class'].T, aggClass.T))
            aggErr = np.multiply(np.sign(aggClass) != np.mat(dataLabel).T, np.ones((sampleNum,1)))
            errRate = aggErr.sum() / sampleNum
            print(str('total err{}\n').format(errRate))
            if errRate == 0.0:
                break
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
            for step in range(-1, sampleNum+1): #这个步长影响大
                for method in ['lt', 'gt']:
                    thres = float(min) + float(step) * stepSize
                    predictClass = self.__caclClass__(dataMat, c_dim, thres, method)
                    errArr = np.mat(np.ones((5, 1))) #numpy 的array和mat 差飞了 不要混合运算
                    errArr[predictClass == labelMat.T] = 0
                    err = errArr.T*weight
                    if bestErr > err:
                        bestErr = err
                        stump['err'] = bestErr
                        stump['dim'] = c_dim
                        stump['ineq'] = method
                        stump['thresh'] = thres
                        stump['class'] = predictClass.copy()
        return stump

    def __caclClass__(self, dataMat, dim, thres, method):
        classLabel = np.ones((dataMat.shape[0], 1))
        if method == 'lt':
            classLabel[dataMat[:,dim] <= thres] = -1
        elif method == 'gt':
            classLabel[dataMat[:,dim] > thres] = -1

        return classLabel
