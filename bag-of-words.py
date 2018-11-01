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
from sklearn.feature_extraction.text import CountVectorizer

path = './txt/'
stemmer = PorterStemmer()
cv = CountVectorizer(max_features=1000, encoding='latin1')

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

        plain_text = " ".join(data)
        return plain_text

names = [f for f in glob.glob(os.path.join(path, '*.txt'))]

dataset = pd.DataFrame({'themes' : [get_class(f) for f in names], 'data' : [get_data(f) for f in names]})

bag_of_words = cv.fit_transform(dataset.data).toarray()