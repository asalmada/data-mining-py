import glob
import os
import re
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

path = './txt/'
stemmer = PorterStemmer()

def get_class (filename):
    return filename.split('-')[0].split('\\')[1]

def get_data (filename):
    with open(filename, 'r', encoding='latin1') as f:
        data = f.read()
        data = re.sub('[^A-Za-z]', ' ', data)
        data = data.lower()
        data = word_tokenize(data)
        
        for token in data:
            if token in stopwords.words('english'):
                data.remove(token)

        for i in range(len(data)):
            data[i] = stemmer.stem(data[i])

        return data

names = [f for f in glob.glob(os.path.join(path, '*.txt'))]

dataset = pd.DataFrame({'journal' : [get_class(f) for f in names], 'data' : [get_data(f) for f in names]})

# Tira caracteres não alfabéticos e deixa o texto inteiro na minúscula  
#dataset.data = dataset.data.map(lambda x: re.sub('[^A-Za-z]', ' ', x).lower())

#dataset.data = pre_processing (dataset.data)

