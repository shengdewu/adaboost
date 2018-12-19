from mlearn import adaboost
import numpy as np

if __name__ == '__main__':
    alg_ada = adaboost()
    dataMat, labelMat = alg_ada.loadSimpData()
    # wight = (np.ones((5,1))/5)
    # bestMethod = alg_ada.buildStump(dataMat, labelMat, wight)
    #print(bestMethod)

    precClass = alg_ada.train(dataMat, labelMat)

    print(precClass)
