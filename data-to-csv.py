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

# Pega o tema do artigo através do nome do arquivo
def get_class (filename):
    return filename.split('-')[0].split('\\')[1]

def get_data (filename):
    with open(filename, 'r', encoding='latin1') as f:

        data = f.read()
        data = re.sub('[^A-Za-z]', ' ', data) # Retira caracteres não alfanumericos
        data = data.lower() # Torna todas as palavras minúsculas

        data = word_tokenize(data) # Remoção das stop words
        for token in data:
            if token in stopwords.words('english'):
                data.remove(token)

        for i in range(len(data)): # Processo de Stemming
            data[i] = stemmer.stem(data[i])

        plain_text = ",".join(data) # Transforma o array de tokens em uma string única separada pro vírgulas
        return plain_text

names = [f for f in glob.glob(os.path.join(path, '*.txt'))]

dataset = pd.DataFrame({'themes' : [get_class(f) for f in names], 'data' : [get_data(f) for f in names]})

dataset.to_csv('data.csv')