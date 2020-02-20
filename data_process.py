from nltk import word_tokenize
from nltk.stem import SnowballStemmer
from os import listdir
from codecs import open as openc
import numpy as np
import heapq

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
def build_bag(f1="ham", f2="spam", saveFile="bag.npy", limit=-1, load=False):
    if(load):
        bag = np.load(saveFile)
    else:
        hamList = getFileList(f1)
        spamList = getFileList(f2)
        bag = []
        # bag = np.empty((0,))
        i = 0

        for file in np.concatenate((hamList, spamList)):
            if(i % 100 == 0):
                print("Building Bag from file: " + str(i) + "/" + str(hamList.shape[0] + spamList.shape[0]))
                # bag = np.unique(bag)

            # bag = np.concatenate((bag, readFileWords(file)))
            bag += readFileWords(file)
            i += 1

        bag = np.unique(bag, return_counts=True)
        np.save(saveFile, bag)

    bag = np.vstack((bag[1], bag[0]), dtype=[('A', 'int'), ('B', 'string')])
    bag = np.sort(bag, axis=1)
    return bag


# def make_representation(fileList, bag):
#     read_file
#
#     for(i in

#
#
# def bagofwords():
#     # for w in sent:
#     # if w not in words:
#     #     ind=word_to_index[w]
#     #     rep[ind]+=1
#     #     #rep += np.eye(vocab_length)[word_to_index[w]]
#     #     words.append(w)


# for sentence in allsentences:        words = word_extraction(sentence)        bag_vector = numpy.zeros(len(vocab))        for w in words:            for i,word in enumerate(vocab):                if word == w:                     bag_vector[i] += 1                            print("{0}\n{1}\n".format(sentence,numpy.array(bag_vector)))

if __name__ == "__main__":
    print(build_bag(load=True, limit=1000))
