import numpy as np
from data_process import importData as importData6
import math

def threshold(ftColumn, Y_ft, numThresh): # (dataSize x 1) np array
    # minN = np.amin(ftColumn)
    # maxN = np.amax(ftColumn)

    # x = list(np.unique(ftColumn))
    # if(minN == maxN):
    #     return (1, 1, True, True) # 1 loss

    # step = int(max((maxN-minN)/numThresh, 1))

    threshRow = []
    lossRow = []
    lType = []
    rType = []

    unqThresh = list(np.unique(ftColumn))
    if(len(unqThresh) <= 1):
        return (1, 1, True, True) # 1 loss

    # for i in range(minN, maxN+1, step):
    for i in unqThresh:
        lDataIndex = np.where(ftColumn < i)[0]
        rDataIndex = np.where(ftColumn >= i)[0]
        lLabels = Y_ft[lDataIndex]
        rLabels = Y_ft[rDataIndex]

        pL = lLabels.shape[0]/(lLabels.shape[0] + rLabels.shape[0])
        pR = 1-pL

        if(pL == 0 or pR == 0):
            continue

        p1L = np.count_nonzero(lLabels)/(lLabels.shape[0])
        p1R = np.count_nonzero(rLabels)/(rLabels.shape[0])
        cL = 1-(p1L**2 + (1-p1L)**2)
        cR = 1-(p1R**2 + (1-p1R)**2)

        cost = (pL * cL) + (pR * cR)
        # print(i, pL, cL, pR, cR, cost)

        threshRow.append(i)
        lossRow.append(cost)
        lType.append( p1L > 0.5 )
        rType.append( p1R > 0.5 )


    minLossIndex = np.argmin(lossRow)
    minLoss = lossRow[minLossIndex]
    minThresh = threshRow[minLossIndex]
    minLType = lType[minLossIndex]
    minRType = rType[minLossIndex]

    return (minLoss, minThresh, minLType, minRType)


def makeTree(X_train, Y_train, treeDepth, numThresh='not implemented'):
    numFeature = X_train.shape[1]
    tree = np.empty((2**treeDepth, 4)).astype('int64') # represent tree as array (using heap rep) height: feature, thresh, leftClassify, rightClassify
    tree[0] = np.array([-1, -1, -1, -1])
    # loop over every treeNode
    for treeIndex in range(1, tree.shape[0]):
    # for treeIndex in range(1, 2):
        # print('building tree node ' + str(treeIndex) + '/' + str(2**treeDepth))

        lossbyFeature = np.empty(numFeature).astype('float64')
        threshbyFeature = np.empty(numFeature).astype('float64')
        leftTypebyFeature = np.empty(numFeature).astype('uint8')
        rightTypebyFeature = np.empty(numFeature).astype('uint8')

        # partition training data based on existing tree
        X_trainft = X_train
        Y_trainft = Y_train

        # partition to correct data assuming existing tree
        div2 = []
        it = treeIndex
        while(it >= 1):
            div2.append(int(it))
            it/=2
        div2.sort()

        for i in div2[:-1]:
            # get condition
            ftLook = int(tree[i,0])
            threshLook = tree[i,1]
            lTypeLook = int(tree[i,2])
            rTypeLook = int(tree[i,3])
            ## CHECK MATH

            if(i*2 in div2):
                partitionIndex = np.where(X_trainft[:,ftLook] < threshLook)
            elif(i*2 + 1 in div2): #odd, right subchild
                partitionIndex = np.where(X_trainft[:,ftLook] >= threshLook)

            X_trainft = X_trainft[partitionIndex]
            Y_trainft = Y_trainft[partitionIndex]
        # print(X_trainft.shape, Y_trainft.shape)


        if(X_trainft.shape[0] == 0): # if no data, do not calculate threshold and use previous threshold
            tree[treeIndex,0] = -1
            tree[treeIndex,1] = -1
            tree[treeIndex,2] = -1
            tree[treeIndex,3] = tree[int(treeIndex/2), 2] if treeIndex % 2 == 0 else tree[int(treeIndex/2), 3]
            continue


        # find partition
        for ftIndex in range(numFeature):
            (lossNum, threshNum, minLType, minRType) = threshold(X_trainft[:,ftIndex], Y_trainft, numThresh)
            # print(ftIndex, lossNum, threshNum, minLType, minRType)
            # if(minLType == minRType):
            #     lossNum = 1
            lossbyFeature[ftIndex] = lossNum
            threshbyFeature[ftIndex] = threshNum
            leftTypebyFeature[ftIndex] = minLType
            rightTypebyFeature[ftIndex] = minRType

        lowFt = np.argmin(lossbyFeature)
        lowThresh = threshbyFeature[lowFt]
        lowLeftType = leftTypebyFeature[lowFt]
        lowRightType = rightTypebyFeature[lowFt]

        tree[treeIndex,0] = lowFt
        tree[treeIndex,1] = lowThresh
        tree[treeIndex,2] = int(lowLeftType)
        tree[treeIndex,3] = int(lowRightType)
        # print(lossbyFeature[lowFt], lowFt, lowThresh, lowLeftType, lowRightType)

    return tree


def testTree(tree, X_test):
    n_test = X_test.shape[0]
    n_feat = X_test.shape[1]
    n_nodes = tree.shape[0]
    y_pred = np.empty(n_test).astype('int8')

    for rowIndex in range(n_test):
        rowPred = -1
        nodeIndex = 1
        while(nodeIndex < tree.shape[0]):
            ft = tree[nodeIndex,0]
            thresh = tree[nodeIndex,1]
            lType = tree[nodeIndex,2]
            rType = tree[nodeIndex,3]

            if(X_test[rowIndex, ft] < thresh):
                nodeIndex = nodeIndex * 2
                rowPred = lType
            else:
                nodeIndex = nodeIndex * 2 + 1
                rowPred = rType

        y_pred[rowIndex] = rowPred

    return y_pred



if __name__ == '__main__':
    (X_train, Y_train, X_test, Y_test) = importData6(bagLimit=10000, returnType='split')
    tree = makeTree(X_train, Y_train, treeDepth=5, numThresh='not implemented')
    y_pred = testTree(tree, X_test)
    print(tree.shape)
    print("\n --> TREE: ")
    print(tree)
    print("\n\n --> ACCURACY:")
    print( (y_pred == Y_test).mean() )
    # print(threshold(X_train[:,335], Y_train, 10))
