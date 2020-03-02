from nltk import word_tokenize
from nltk.stem import SnowballStemmer
from os import listdir
from codecs import open as openc
import numpy as np
import heapq

from bayes import naive_bayes
from knn import knn

# @returns list of files
def getFileList(folder):
    fileList = np.empty((0,))
    for file in listdir(folder):
        if(file.endswith(".txt")):
            fileList = np.append(fileList, folder + "/" + file)
    return fileList



# @return list of stemmed/processed words for a file
stemmer = SnowballStemmer("english", ignore_stopwords=False)

def readFileWords(file):
    f = openc(file, encoding='latin-1')
    return [stemmer.stem(word) for word in word_tokenize(f.read()) if word.isalpha()]



# @return bag of words
def build_bag(fileList, saveFile="bag.npy", limit=-1, load=False):
    if(load):
        bag = np.load(saveFile)
    else:
        bag = []
        # bag = np.empty((0,))
        i = 0

        for file in fileList:
            if(i % 100 == 0):
                print("Building Bag from file: " + str(i) + "/" + str(fileList.shape[0]))
                # bag = np.unique(bag)

            # bag = np.concatenate((bag, readFileWords(file)))
            bag += readFileWords(file)
            i += 1

        bag = np.unique(bag, return_counts=True)
        np.save(saveFile, bag)

    (bag, cnt) = bag

    # sort
    cnt = cnt.astype('int64')
    cntSort = np.flip(np.argsort(cnt))
    bag = bag[cntSort]

    return np.sort(bag[:limit])




def convertTextBag(fileList, bag, saveFile, load=False):
    if(load):
        vectors = np.load(saveFile)
    else:
        vectors = np.empty((fileList.shape[0], bag.shape[0]), dtype='int16')

        for fileNumber in range(fileList.shape[0]):
            if(fileNumber % 100 == 0):
                print("Converting to Vector " + saveFile + ": " + str(fileNumber) + "/" + str(fileList.shape[0]))

            currentVector = np.empty((1, bag.shape[0]), dtype='int16')
            wordList = np.array(readFileWords(fileList[fileNumber]))

            for i in range(bag.shape[0]):
                currentVector[0,i] = np.where(wordList==bag[i])[0].shape[0]

            vectors[fileNumber,:] = currentVector

        np.save(saveFile, vectors)

    return vectors


def importData(bagLimit=-1, loadVectors=True, loadBag=True, returnType='normal', splitPercentage=0.7):
    if(loadVectors):
        hamVector = np.load("ham" + str(bagLimit) + ".npy")
        spamVector = np.load("spam" + str(bagLimit) + ".npy")
    else:
        hamFiles = getFileList("ham")
        spamFiles = getFileList("spam")
        allFiles = np.concatenate((hamFiles, spamFiles))
        bag = build_bag(allFiles, load=loadBag, limit=bagLimit)

        hamVector = convertTextBag(hamFiles, bag, saveFile="ham" + str(bagLimit))
        spamVector = convertTextBag(spamFiles, bag, saveFile="spam" + str(bagLimit))

    if(returnType == 'class'):
        return (hamVector, spamVector)
    elif(returnType == 'split'): #@return X_train, Y_train, X_test, Y_test
        X = np.vstack((hamVector, spamVector))
        Y = np.concatenate(( np.zeros((hamVector.shape[0])), np.ones((spamVector.shape[0])) ))
        rand = np.random.permutation(X.shape[0])
        X = X[rand]
        Y = Y[rand]
        splitN = int(X.shape[0]*splitPercentage)
        return (X[:splitN].astype('int32'), Y[:splitN].astype('uint8'), X[splitN:].astype('int32'), Y[splitN:].astype('uint8'))
    else: #@return train, test
        raise NotImplementedError()

        # for bagword in bag:


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = importData(bagLimit=5000, returnType='split')
    y_pred = naive_bayes(X_train, Y_train, X_test)
    # y_pred = knn(X_train, Y_train, X_test, k=5, L='L2')

    # print(y_pred)
    print((y_pred == Y_test).mean())
