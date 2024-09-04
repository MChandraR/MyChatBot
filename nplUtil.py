import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

psteam = PorterStemmer()

def tokenize(words):
    return [w for w in words if w != " "]
    #return nltk.word_tokenize(words)

def stemm(word):
    return psteam.stem(word.lower())

def bag(wordlist,words):
    words = [stemm(w) for w in words]
    
    bag = np.zeros(len(wordlist),dtype=np.float32)
    for idx,w in enumerate(wordlist):
        if w in words:
            bag[idx] = 1.0
    
    return bag
    
