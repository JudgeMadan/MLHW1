from nltk import word_tokenize
from nltk.stem import SnowballStemmer
import numpy as np


def stemming(input):
    stemmer = SnowballStemmer("english", ignore_stopwords=False)
    # return [porter.stem(word) for word in word_tokenize(input)]


def bagofwords():
    # for w in sent:
    # if w not in words:
    #     ind=word_to_index[w]
    #     rep[ind]+=1
    #     #rep += np.eye(vocab_length)[word_to_index[w]]
    #     words.append(w)
